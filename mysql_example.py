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

def create_connection(host_name, user_name, user_password):
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            user=user_name,
            passwd=user_password
        )
        print("Connection to MySQL DB successful")

    except Error as e:
        print(f"The error '{e}' occurred")

    return connection

connection = create_connection("localhost", "root", "")