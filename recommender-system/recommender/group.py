import pandas as pd

#TODO: Check duplicities
"""
    Module in charge of group management. A group is a collection of users
    that has a name and a context type assigned to it.

    Modulo encargado de la gestión de grupos. Un grupo es una colección de usuarios
    que tiene asignado un nombre y un tipo de contexto
"""

class Group():
    def __init__(
        self, group_name="default", group_context_name="none",
        verbose=True
    ):
        """
        Constructor
            :param group_name: nombre del grupo
            :type string:

            :param group_context_name: nombre del contexto
            :type string:
            
        """
        self.module = "Group: "

        self.group_name = group_name
        self.group_context_name = group_context_name

        self.users = []

        self.verbose = verbose

    def add_user_id(self, user_id):
        '''
            Adds the given user to the group (number in string format)
        '''
        #TODO Comprobar duplicados
        #TODO Check if user exists first
        #TODO Comprobar que es un numero in string format
        if self.verbose : print(self.module + " Adding user to the group: "+user_id)
        self.users.append(user_id)

    def add_list_of_users(self, user_list):
        '''
            Adds various users to the group (array of numbers in string format)
        '''
        #TODO Comprobar duplicados
        #TODO Check if user exists first
        #TODO Check the format
        if self.verbose : print(self.module + " Adding "+str(len(user_list))+" users to the group")
        self.users = self.users + user_list

    def print_users(self, simple=False, group_details=True):
        '''
            Prints the group information
        '''
        if group_details:
            print(self.module + " Group "+self.group_name+" context: " + self.group_context_name)
        if self.is_empty_group():
            print(self.module + " The group is empty")
        else:
            print(self.module + " The group has "+str(len(self.users))+" members:")
        if not simple:
            i = 0
            for user in self.users:
                i+=1
                print("\t"+str(i)+") "+user)

    def is_empty_group(self):
        '''
            Checks if is an empty group
        '''
        if len(self.users) == 0:
            return True
        else:
            return False