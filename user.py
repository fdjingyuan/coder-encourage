from flask_login import UserMixin
from DatabaseConnection import exec_fetch_user_with_id, database


class User(UserMixin):
    def __init__(self, username, id):
        self.username = username
        self.id = id


    @staticmethod
    def get(user_id):
        """try to return user_id corresponding User object.
        This method is used by load_user callback function
        """
        if not user_id:
            return None
        else:
            database.connect_database()
            user_info = exec_fetch_user_with_id(int(user_id))
            database.disconnect_database()

        if user_info is not None:
            return User(id=user_info[0], username=user_info[1])
        else:
            return None

