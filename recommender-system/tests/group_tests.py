import sys
sys.path.append("../")

from recommender.group import Group

from configuration import *

print("Group Module testing script.")

user = cfg["custom_user"]
custom_group = cfg['example_group']

test_n = 0
successful_tests = 0

test_n = new_tests(test_n, "Creación de un grupo y adesión de un único usuario y después varios.")
try:
    group = Group(group_name="Test "+str(test_n), group_context_name="Random")
    group.print_users()
    group.add_user_id(user)
    group.print_users()
    group.add_list_of_users(custom_group)
    group.print_users()
    successful_tests += 1
except Exception as ex:
    print(ex)

test_n = new_tests(test_n, "Creación de un grupo y adesión de varios y después uno.")
try:
    group = Group(group_name="Test "+str(test_n), group_context_name="Random")
    group.print_users()
    group.add_user_id(user)
    group.print_users()
    group.add_list_of_users(custom_group)
    group.print_users()
    successful_tests += 1
except Exception as ex:
    print(ex)

successfull_tests_print(successful_tests, test_n)