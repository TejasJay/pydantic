import mysql.connector
from mysql.connector import Error
from dataclasses import dataclass
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from typing import List, Dict, Any
import asyncio
from load_models import MODEL
from pydantic_ai.models.google import GoogleModelSettings
import json

settings = GoogleModelSettings(google_thinking_config={'include_thoughts': True})


file_path = 'netflix_agent/netflixdb-mysql.sql'

host = '127.0.0.1'
port = 3306
user = 'mysql'
password = 'mysql'
database = 'netflixdb'

# --- END OF CONFIGURATION ---





def import_sql_file(host, user, password, database, file_path, port=3306):
    """
    Connects to a MySQL database and executes all SQL commands from a given .sql file.

    Args:
        host (str): The database server host (e.g., '127.0.0.1').
        user (str): The database username.
        password (str): The database user's password.
        database (str): The name of the database to connect to.
        file_path (str): The full path to the .sql file to be executed.
        port (int): The port number for the database connection.
    """
    connection = None
    cursor = None
    try:
        # Establish the database connection
        connection = mysql.connector.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database
        )
        cursor = connection.cursor()
        print(f"Successfully connected to database '{database}'")

        # Open and read the SQL file
        print(f"Reading SQL file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as sql_file:
            # Split the file into individual SQL statements
            # The delimiter is ';' followed by a newline.
            # This is a simple approach and works for most standard SQL dump files.
            sql_commands = sql_file.read().split(';\n')

        # Execute every command from the file
        print("Executing SQL script...")
        command_count = 0
        for command in sql_commands:
            # Skip empty commands
            if command.strip():
                try:
                    cursor.execute(command)
                    command_count += 1
                except Error as e:
                    print(f"Error executing command: {e}")
                    # Optional: decide if you want to stop on error or continue
                    # raise  # Uncomment to stop the script on the first error

        # Commit the changes
        connection.commit()
        print(f"Successfully executed {command_count} SQL commands.")
        print("Database import completed successfully.")

    except Error as e:
        print(f"Error while connecting to MySQL or executing script: {e}")
    finally:
        # Close the connection
        if connection and connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed.")


# import_sql_file(host, user, password, database, file_path, port=3306)



@dataclass
class DatabaseDependencies:
    db_connection: mysql.connector.MySQLConnection



class TableSchema(BaseModel):
    table_name: str = Field(description="The name of the database table.")
    columns: list[str] = Field(description="The list of column names in the table.")



class Output(BaseModel):
    question: str = Field(description='The question asked by the user')
    answer: str | None = Field(description='The answer returned by the Agent')
    query: str = Field(description='The query used by the agent to execute on the DB')
    tables: list[str] = Field(description='The tables listed by the list_tables tool')
    columns: list[TableSchema] = Field(description='The schemas for the tables used in the query')
    thought: str = Field(description='The thought process of the agent for every single tool it uses.')




agent = Agent(
    system_prompt="You are a helpful database assistant. You can query the netflixdb database to answer questions. Make sure that the Title or Name is never empty. First, use the `list_tables` tool to see what tables are available. Then, if you need to know the columns of the tables to construct a query, use the `get_table_schemas` tool with the relevant table names. Finally, use the `execute_sql_query` tool to answer the user's question. List only the top 5 results always.",
    deps_type=DatabaseDependencies,
    model=MODEL,
    output_type=Output,
    model_settings=settings
)


@agent.tool
async def list_tables(ctx: RunContext[DatabaseDependencies]) -> List[str]:
    """Lists all the tables in the database."""
    cursor = ctx.deps.db_connection.cursor()
    cursor.execute("SHOW TABLES;")
    tables = [table[0] for table in cursor.fetchall()]
    cursor.close()
    return tables

@agent.tool
async def get_table_schemas(ctx: RunContext[DatabaseDependencies], tables: List[str]) -> Dict[str, Any]:
    """Gets the schema (column names) for a given list of tables."""
    cursor = ctx.deps.db_connection.cursor()
    schemas = {}
    for table in tables:
        try:
            cursor.execute(f"DESCRIBE {table};")
            columns = [row[0] for row in cursor.fetchall()]
            schemas[table] = columns
        except mysql.connector.Error as err:
            schemas[table] = {"error": str(err)}
    cursor.close()
    return schemas


@agent.tool
async def execute_sql_query(ctx: RunContext[DatabaseDependencies], query: str) -> Dict[str, Any]:
    """Executes a SQL query and returns the result."""
    cursor = ctx.deps.db_connection.cursor(dictionary=True)
    try:
        cursor.execute(query)
        result = cursor.fetchall()
        ctx.deps.db_connection.commit()
        return {"result": result}
    except mysql.connector.Error as err:
        return {"error": str(err)}
    finally:
        cursor.close()


async def main():
    cnx = None
    try:
        cnx = mysql.connector.connect(
            host="127.0.0.1",
            port=3306,
            user="mysql",
            password="mysql",
            database='netflixdb'
        )
        print("\nConnection to MySQL database successful")

        deps = DatabaseDependencies(db_connection=cnx)

        result = await agent.run(
            ["Give me top 5 the best tv shows names, that has the highest ratings, and highest number of views. I want to see the movies, date it was released, rating, views."],
            deps=deps
        )
        # print('\n')
        # print(result.all_messages(), end='\n')
        print('\n')
        print("Agent Response:")
        print(result.output.answer)

        print('\n')
        print('Complete Pydantic JSON:')
        output_dict = result.output.model_dump()

        print(json.dumps(output_dict, indent=4))


    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        if cnx and cnx.is_connected():
            cnx.close()
            print("\nMySQL connection is closed")

if __name__ == '__main__':
    asyncio.run(main())