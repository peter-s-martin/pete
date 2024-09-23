from pyspark.conf import SparkConf
from pyspark.context import SparkContext
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import col, lower, concat, lit, max, hash,udf, when, sum, row_number
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, ArrayType, DoubleType
from pyspark.ml.feature import BucketedRandomProjectionLSH,StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.linalg import Vectors, VectorUDT
import networkx as nx

# Initialize Spark session
spark = SparkSession.builder.appName("Householding").getOrCreate()
spark.catalog.clearCache()

# Load member data from the external table
members_df = spark.sql('''
    SELECT  
        m.memberid as member_id,
        m.personname,
        concat(nvl(PERMANENTSTREETNAME,''),'',nvl(PERMANENTSTREETNUMBER,''),'default',nvl(PERMANENTCITY,''),'',nvl(PERMANENTZIPCODE,''),'',nvl(PERMANENTCOUNTRY,'')) as address,
        coalesce(PHONENUMBERMOBILE,PHONENUMBERWORK,PHONENUMBERHOME,'0') as phone,
        coalesce(EMAILADDRESS,'default') as email,
        coalesce(TAXIDENTIFICATIONNUMBER,0) as tax_id,
        --depositacctcount as total_dep_acct,
        --loansacctcount as total_loan_acct,
        --depositacctendingbalance as dep_acct_balance,
        --loansacctendingbalance as loan_acct_balance,
        coalesce(depositacctcount+loansacctcount,0) as CountofAccounts,
        coalesce(depositacctendingbalance,0) as totaldepositbalance,
        coalesce(membersincedateid,current_date()) as join_date,
        current_date() as eff_dt,
        0 as weight
    FROM CZ_EDW.FCT_MEMBER_CURATED_MONTHLY M
    --WHERE
    --DATE_FORMAT(MONTHID,'yyyy-MM') = DATE_FORMAT(CURRENT_TIMESTAMP,'yyyy-MM')
''')

# Member Relationship Type DF
member_rel_type_df = spark.read.table('lz_ff.member_rel_type')

# Vector Index list identification column list
#** Column name should match with member_df
vect_index_list = ['address','phone','email','tax_id']
vector_column_list=[]

# Get the maximum eff_dt from the relationships DataFrame
current_dt = members_df.select(max("join_date")).first()[0]

members=members_df
members.show(truncate=False)

# Feature extraction

for indexlist in vect_index_list:
    # Index of the column
    members_df = StringIndexer(inputCol=indexlist, outputCol=indexlist+"_index")\
                .setHandleInvalid("keep").fit(members_df).transform(members_df)

    # Encode of the index
    members_df = OneHotEncoder(inputCol=indexlist+"_index", outputCol=indexlist+"_vec") \
                    .setHandleInvalid("keep").fit(members_df).transform(members_df)
    # combining all the vector column index list  
    vector_column_list.append(indexlist+"_vec")
                 

# Assemble the feature vectors
assembler = VectorAssembler(inputCols=vector_column_list, outputCol="features")
members_df = assembler.transform(members_df)

# Show the DataFrame with feature vectors
#print("DataFrame with feature vectors:")

# Create vertices DataFrame
vertices = members_df.select(col("member_id").alias("member_id"))

#vertices.show(truncate=False)

# Create edges DataFrame based on address and phone similarity
edges = members_df.alias("df1").join(members_df.alias("df2"), col("df1.address") == col("df2.address")) \
    .select(col("df1.member_id").alias("parent_member_id"), col("df2.member_id").alias("child_member_id")) \
    .withColumn('rel_code',lit(2)).distinct()

#edges.show(truncate=False)

member_rel_df = edges.join(member_rel_type_df,edges.rel_code == member_rel_type_df.rel_cd,"inner") \
    .select(
    edges["*"],
    col("rel_directional_flag"),
    when((col("is_current") == 1) & (col("householding_flg") == 1), col("weight")).otherwise(0).alias("weight")
)
   
#member_rel_df.show(truncate=False)

# Aggregate the weights for multiple edges between the same parent and child
edges_aggregated = member_rel_df.groupBy("parent_member_id","child_member_id").agg(sum("weight").alias("total_weight"))  

#edges_aggregated.show(truncate=False)

# Filter edges with total weight >= 1
filtered_edges = edges_aggregated.filter(col("total_weight") >= 1)

#filtered_edges.show(truncate=False)
members_df.show(5)

# Create the NetworkX graph
G = nx.DiGraph()

# Add vertices (nodes)
for row in members_df.collect():
    G.add_node(row['member_id'])

# Add edges to the graph with weights
for row in filtered_edges.collect():
    G.add_edge(row['parent_member_id'], row['child_member_id'], weight=row['total_weight'])
   
# Find strongly connected components (for directed graphs)
strongly_connected_components = list(nx.strongly_connected_components(G))

print(strongly_connected_components)

# Assign unique household IDs and collect members
household_members = []
for i, component in enumerate(strongly_connected_components,start=1):
    for member in component:
        household_members.append((i, int(member)))


# Convert to Spark DataFrame
household_members_df = spark.createDataFrame(household_members, schema=["household_id", "member_id"])

household_members_df.show()
members.show()

# Add `eff_dt` to `household_members_df`
household_members_df = household_members_df.withColumn("eff_dt", lit(current_dt))

# Join household_members_df with members to get additional member info
household_members = household_members_df.join(members.select("member_id", "eff_dt", "CountofAccounts", "totaldepositbalance", "join_date"), on=["member_id"], how="left")

household_members.show()

# Define window specification to rank members within each household
window_spec = Window.partitionBy("household_id").orderBy(
    col("CountofAccounts").desc(),
    col("totaldepositbalance").desc(),
    col("join_date").asc(),
    col("member_id").asc()
)

# Add row number based on ranking criteria
household_members = household_members.withColumn("rank", row_number().over(window_spec))

# Select the head of household (hoh_member_id) for each household
hoh_members = household_members.filter(col("rank") == 1).select("household_id", col("member_id").alias("hoh_member_id"))

# Join the head of household back with the household_members DataFrame
household_members_with_hoh = household_members.join(hoh_members, on="household_id", how="left")

household_members_with_hoh.show(5)

household_members_with_hoh.createOrReplaceTempView('household_members')

### DIM_PARTY_HOUSEHOLD Table Load
## Create Temp Table
df_stg_hh = spark.sql(f'''
Select
household_id as HOUSEHOLDID,
HPD.PARTYID AS HEADOFHOUSEHOLDPARTYID,
PD.PARTYID AS HOUSEHOLDPARTYID,
current_timestamp() as HOUSEHOLDSTARTDATE,
NULL as HOUSEHOLDENDDATE,
'Y' as CURRENTRECORDYN
FROM
household_members hm
LEFT JOIN sz_edw.dim_party pd on ( pd.SOURCEPARTYID=hm.member_id AND pd.CURRENTRECORDYN='Y' AND pd.ISDELETED='N')
LEFT JOIN sz_edw.dim_party hpd on ( hpd.SOURCEPARTYID=hm.hoh_member_id AND hpd.CURRENTRECORDYN='Y' AND hpd.ISDELETED='N')
''')
#df_stg_appl.show(5, truncate=False)
df_stg_hh.createOrReplaceTempView('df_stg_hh')
