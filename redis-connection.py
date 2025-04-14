import os
import redis

# Redis connection parameters
# if os.environ["ENV_EXE"] == "prod":
#     host = 'prod.mce-vecotr-db-redis.dx0ahi.use1.cache.amazonaws.com'
# else:
# host = 'cmdb-service-065e5da12eafa418.elb.us-east-1.amazonaws.com'
host = 'master.mce-vecotr-db-redis.dx0ahi.use1.cache.amazonaws.com'
port = 6379
tls = True  # Enable TLS

# Create a connection to the Redis instance
def redis_connection():
    r = redis.StrictRedis(
        host=host,
        port=port,
        ssl=tls,
        ssl_certfile=None,  # Add your certificate file here if needed
        ssl_keyfile=None,   # Add your private key file here if needed
        ssl_ca_certs=None,  # Add CA certificates here if needed
    )

    # Test the connection
    try:
        response = r.ping()  # Test if the connection is successful
        print("Connected to Redis:", response)
    except redis.ConnectionError as e:
        print("Connection failed:", e)




if __name__ == "__main__":
    redis_connection()