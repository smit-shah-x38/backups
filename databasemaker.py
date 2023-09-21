import weaviate

auth_config = weaviate.AuthApiKey(api_key="OODEr9AZNBadP9VxHqac67HfiZ0pFCMC5wYr")

client = weaviate.Client(
    url="https://testcluster-d74yev6k.weaviate.network", auth_client_secret=auth_config
)

# Define a schema for a collection of movies and actors
schema = {
    "classes": [
        {
            "class": "Movie",
            "description": "A movie",
            "properties": [
                {
                    "name": "title",
                    "description": "The title of the movie",
                    "dataType": ["text"],
                },
                {
                    "name": "genre",
                    "description": "The genre of the movie",
                    "dataType": ["text"],
                },
                {
                    "name": "year",
                    "description": "The year of release of the movie",
                    "dataType": ["int"],
                },
                {
                    "name": "cast",
                    "description": "The actors who starred in the movie",
                    "dataType": ["Actor"],
                },
            ],
        },
        {
            "class": "Actor",
            "description": "An actor",
            "properties": [
                {
                    "name": "name",
                    "description": "The name of the actor",
                    "dataType": ["text"],
                },
                {
                    "name": "age",
                    "description": "The age of the actor",
                    "dataType": ["int"],
                },
                {
                    "name": "movies",
                    "description": "The movies that the actor starred in",
                    "dataType": ["Movie"],
                },
            ],
        },
    ]
}

# Create the schema on Weaviate
client.schema.create(schema)

# Import some data into the collection
# Note: You can also use batch import or CSV import methods
client.data_object.create(
    {
        # The class of the data object
        "@class": "Movie",
        # The properties of the data object
        # Note: You can also specify custom vectors or IDs
        "@properties": {
            # A text property
            # Note: You can also use transformers or validators
            # to process or validate the text values
            # See https://www.semi.technology/documentation/weaviate/current/schema/text.html
            # for more details
            "title": {"@value": "The Matrix"}
        },
    }
)
# A text property with multiple values
# Note: You can also use transformers or validators
# to process or validate the text values
# See https://www.semi.technology/documentation/weaviate/current/schema/text.html
# for more details
