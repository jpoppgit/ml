#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
    Practical example of MySQL.

    Inspired by
    https://realpython.com/python-sql-libraries/

    MySQL server
      - installation (macOS): brew install mysql
      - steering
        - brew services start mysql
        - brew services stop mysql
        - brew services restart mysql
        - connect: mysql -u root
"""

import mysql.connector
from mysql.connector import Error

import logging

logging.basicConfig(level=logging.INFO)

def create_database(connection, query):
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        logging.info("Database created successfully")
    except Error as e:
        logging.info(f"The error '{e}' occurred")


def create_connection_initial(host_name, user_name, user_password):
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            user=user_name,
            passwd=user_password
        )
        logging.info("Connection to MySQL DB successful")

    except Error as e:
        logging.info(f"The error '{e}' occurred")

    return connection

def create_connection(host_name, user_name, user_password, db_name):
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            user=user_name,
            passwd=user_password,
            database=db_name
        )
        logging.info("Connection to MySQL DB successful")

    except Error as e:
        logging.info(f"The error '{e}' occurred")

    return connection

def execute_query(connection, query):
    cursor = connection.cursor()
    logging.info(query)
    try:
        cursor.execute(query)
        connection.commit()
        logging.info("Query executed successfully")

    except Error as e:
        logging.info(f"The error '{e}' occurred")

def execute_read_query(connection, query):
    logging.info(query)
    cursor = connection.cursor()
    result = None
    try:
        cursor.execute(query)
        result = cursor.fetchall()
        return result
    except Error as e:
        logging.info(f"The error '{e}' occurred")


# ----------------------------
# MySQL interactions
# ----------------------------
db_name = 'sm_app'

# -- initially creates DB -----
#connection = create_connection_initial("localhost", "root", "")
#create_database_query = "CREATE DATABASE "+db_name
#create_database(connection, create_database_query)

# -- creates tables -----
# connection = create_connection("localhost", "root", "", db_name)

# create_users_table = """
# CREATE TABLE IF NOT EXISTS users (
#   id INT AUTO_INCREMENT, 
#   name TEXT NOT NULL, 
#   age INT, 
#   gender TEXT, 
#   nationality TEXT, 
#   PRIMARY KEY (id)
# ) ENGINE = InnoDB
# """
# execute_query(connection, create_users_table)

# create_posts_table = """
# CREATE TABLE IF NOT EXISTS posts (
#   id INT AUTO_INCREMENT, 
#   title TEXT NOT NULL, 
#   description TEXT NOT NULL, 
#   user_id INTEGER NOT NULL, 
#   FOREIGN KEY fk_user_id (user_id) REFERENCES users(id), 
#   PRIMARY KEY (id)
# ) ENGINE = InnoDB
# """
# execute_query(connection, create_posts_table)

# -- inserts records -----
# connection = create_connection("localhost", "root", "", db_name)

# create_users = """
# INSERT INTO
#   `users` (`name`, `age`, `gender`, `nationality`)
# VALUES
#   ('James', 25, 'male', 'USA'),
#   ('Leila', 32, 'female', 'France'),
#   ('Brigitte', 35, 'female', 'England'),
#   ('Mike', 40, 'male', 'Denmark'),
#   ('Elizabeth', 21, 'female', 'Canada');
# """
# execute_query(connection, create_users)

# -- selects records -----
# connection = create_connection("localhost", "root", "", db_name)

# select_users = "SELECT * FROM users"
# users = execute_read_query(connection, select_users)
# for user in users:
#     logging.info(user)
