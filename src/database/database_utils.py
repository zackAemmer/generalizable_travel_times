import pandas as pd
import psycopg2

def execute_query(conn, query):
    cur = conn.cursor()
    cur.execute(query)
    return cur.fetchall()

def get_adjacent_shingles(conn, dist_deg, time_s, bearing):
    query = f"\
    WITH t1 AS (SELECT * \
    FROM shingles) \
    SELECT * \
    FROM t1 JOIN shingles ON ST_DWithin(t1.geom, shingles.geom, {dist_deg}) AND t1.file = shingles.file AND t1.shingle_id != shingles.shingle_id AND (t1.timeID_s - shingles.timeID_s) >= 0 AND (t1.timeID_s - shingles.timeID_s) <= {time_s} AND abs(t1.bearing - shingles.bearing) <= {bearing}\
    "
    cur = conn.cursor()
    cur.execute(query)
    rows=cur.fetchall()
    cols=[desc[0] for desc in cur.description]
    adjacent_shingles = pd.DataFrame(rows, columns=cols)
    adjacent_shingles = adjacent_shingles[['shingle_id','timeid_s']].copy()
    adjacent_shingles.columns = ['pred_shingle','shingle','pred_timeid_s','timeid_s']
    adjacent_shingles = adjacent_shingles.astype(dtype={"pred_shingle":"int64", "shingle":"int64", "pred_timeid_s":"int64", "timeid_s":"int64"})
    return adjacent_shingles

def overwrite_shingles_table(conn, df):
    """
    Drops shingles table and uploads new set from df.
    Expect roughly 1.5 minutes per 1 million points.
    """
    # Reset table
    cur = conn.cursor()
    query = "DROP TABLE IF EXISTS shingles;"
    cur.execute(query)
    query = "CREATE TABLE shingles ( trip_id VARCHAR(64), file VARCHAR(16), shingle_id VARCHAR(64), lon NUMERIC(13,10), lat NUMERIC(13,10), timeID_s NUMERIC(10), bearing NUMERIC(4,1));"
    cur.execute(query)
    # Load the new data into the shingles table
    df = df[['trip_id','file','shingle_id','lon','lat','timeID_s','bearing']]
    execute_mogrify(conn, df, "shingles")
    # Add geometry and index to shingles table
    cur.execute(f"ALTER TABLE shingles ADD COLUMN geom geometry(Point, 4326);")
    conn.commit()
    cur.execute(f"UPDATE shingles SET geom = ST_SetSRID(ST_MakePoint(lon, lat), 4326);")
    conn.commit()
    cur.execute(f"CREATE INDEX shingle_geom_idx ON shingles USING GIST (geom);")
    conn.commit()
    cur.close()
    return None

def execute_mogrify(conn, df, table):
    """
    Using cursor.mogrify() to build the bulk insert query
    then cursor.execute() to execute the query
    """
    # Create a list of tuples from the dataframe values
    tuples = [tuple(x) for x in df.to_numpy()]
    # Comma-separated dataframe columns
    cols = ','.join(list(df.columns))
    # SQL quert to execute
    cursor = conn.cursor()
    values = [cursor.mogrify("(%s,%s,%s,%s,%s,%s,%s)", tup).decode('utf8') for tup in tuples]
    query  = "INSERT INTO %s(%s) VALUES " % (table, cols) + ",".join(values)
    try:
        cursor.execute(query, tuples)
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error: %s" % error)
        conn.rollback()
        cursor.close()
        return 1
    print("execute_mogrify() done")
    cursor.close()