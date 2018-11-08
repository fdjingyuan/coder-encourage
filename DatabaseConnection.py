import pymysql
import dbconfig


class Connection(object):
    def connect_database(self):
        self.connection = pymysql.connect(host=dbconfig.host,
                                          user=dbconfig.user,
                                          db=dbconfig.db,
                                          passwd=dbconfig.passwd,
                                          port=dbconfig.port,
                                          charset=dbconfig.charset
                                          )
        self.cursor = self.connection.cursor()

    def disconnect_database(self):
        self.cursor.close()
        self.connection.close()

    def exec_query(self, sql):
        self.cursor.execute(sql)

    def exec_update(self, sql):
        self.cursor.execute(sql)
        self.connection.commit()

    def fetch_cursor(self):
        return self.cursor.fetchall()


database = Connection()


def exec_user_login(username, password):
    sql = "SELECT user_id, user_permission FROM users WHERE user_name='%s' AND user_password='%s'"
    database.exec_query(sql % (username, password))
    user = database.fetch_cursor()
    if len(user) == 0:
        return None, None
    else:
        return user[0][0], user[0][1]


def exec_fetch_user_with_id(id):
    sql = "SELECT * FROM users WHERE user_id='%d'"
    database.exec_query(sql % id)
    user = database.fetch_cursor()
    if len(user) == 0:
        return None
    else:
        return user[0]