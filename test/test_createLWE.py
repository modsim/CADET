import os
import shutil
import subprocess
from itertools import combinations

import numpy as np
from cadet import Cadet
from deepdiff import DeepDiff
from deepdiff.model import PrettyOrderedSet
from git import Repo


def create_arguments_list():
    """
    Create a list of command-line arguments for the createLWE.exe executable.

    Returns:
    list: A list of argument strings.
    """
    arguments_list = []
    args = {
        '-t': [5.0],
        '-T': [1800.0],
        '-c': [1.0],
        '-s': [0.1],
        '--rad': [5],
        '--parTypes': [2],
        "-u": ["LRMP", "LRM", "GRM2D"],
        "--par": [5],
        "--col": [11],
    }

    boolean_args = [
        "--solverTimes",
        "--reverseFlow",
        "--ad",
        "--kinetic",
    ]

    for key, values in args.items():
        for value in values:
            arguments_list.append(f"{key} {value}")

    for boolean_arg in boolean_args:
        arguments_list.append(boolean_arg)

    return arguments_list


def checkout_and_pull_branch(repo, branch):
    """
    Checkout and pull the latest changes for a given branch.

    Parameters:
    branch (str): The branch to checkout and pull.
    """
    repo.git.checkout(branch)
    repo.git.pull()


def filepath_from_branch_and_argument(branch, argument):
    """
    Generate the file path for the LWE file based on the branch and argument.

    Parameters:
    branch (str): The branch the file belongs to.
    argument (str): The argument string used to generate the file.

    Returns:
    str: The generated file path for the LWE file.
    """
    return f"._tmp/{branch}/lwe_{argument.replace('-', '').replace(' ', '_')}.h5"


def run_create_lwe(branch, argument):
    """
    Run the createLWE executable with specified arguments and save the output to temporary files.

    Parameters:
    branch (str): The branch being tested.
    arguments_list (list): String of arguments for the createLWE executable.
    """
    filepath = filepath_from_branch_and_argument(branch, argument)
    results_lwe = subprocess.run(
        args=f"._tmp/{branch}/createLWE.exe -o {filepath} {argument}",
        capture_output=True
    )
    if results_lwe.returncode != 0:
        print(results_lwe.stdout.decode(), results_lwe.stderr.decode())


def load_lwe_file(branch, argument):
    """
    Load an LWE file into a Cadet simulation object.

    Parameters:
    branch (str): The branch the file belongs to.
    argument (str): The argument string used to generate the file.

    Returns:
    Cadet: A Cadet simulation object.
    """
    sim = Cadet()
    sim.filename = filepath_from_branch_and_argument(branch, argument)
    sim.load()
    return sim


def are_dicts_equal(dict1, dict2):
    """
    Compares two nested dictionaries, including numpy arrays.

    Parameters:
    dict1 (dict): First dictionary to compare.
    dict2 (dict): Second dictionary to compare.

    Returns:
    bool: True if all keys and values are identical, False otherwise.
    """
    if dict1.keys() != dict2.keys():
        return False

    for key in dict1:
        value1 = dict1[key]
        value2 = dict2[key]

        if isinstance(value1, dict) and isinstance(value2, dict):
            if not are_dicts_equal(value1, value2):
                print("Dictionary unequal", key, "\n", value1, "\n", value2)
                return False
        elif isinstance(value1, np.ndarray) and isinstance(value2, np.ndarray):
            if key == "connections":
                continue
            if not np.array_equal(value1, value2):
                print("Array unequal", key, "\n", value1, "\n", value2)
                return False
        elif value1 != value2:
            print("Value unequal", key, value1, value2)
            return False

    return True


def pretty_print(diff, prefix=""):
    """
    Recursively prints a nested dictionary in a readable format.

    Parameters:
    diff (dict): The nested dictionary to print.
    prefix (str): A string prefix for indentation (default is an empty string).
    """
    for key, value in diff.items():
        if type(value) is dict:
            print(f"{prefix}{key}:", "{")
            pretty_print(value, prefix=prefix + "\t")
            print(f"{prefix}", "}")
        elif type(value) is PrettyOrderedSet:
            print(f"{prefix}{key}:", "{")
            print(prefix + "\t" + f"\n{prefix}\t".join(value))
            print(f"{prefix}", "}")
        else:
            print(f"{prefix}{key}: {value}")


def compare_across_branches(arguments_list, branches):
    """
    Compare simulation results across branches for each argument.

    Parameters:
    arguments_list (list): List of argument strings.
    """
    for branch1, branch2 in combinations(branches, 2):
        for argument in arguments_list:
            print(f"Changes from branch '{branch1}' to '{branch2}' with arguments '{argument}':")
            sim1 = load_lwe_file(branch1, argument)
            sim2 = load_lwe_file(branch2, argument)
            diff = DeepDiff(sim1.root, sim2.root)
            pretty_print(diff, prefix="\t")

    return


def build_cadet():
    results_build = subprocess.run("test\\build_windows.bat", capture_output=True)
    if results_build.returncode != 0:
        print(results_build.stdout.decode(), results_build.stderr.decode())
    return


def main():
    # Change to the parent directory
    os.chdir(r"C:\Users\ronal\Documents\CADET")

    # Initialize the repository and switch to the specified branches
    repo = Repo(".")
    branches = ["master", "fix/createLWE"]

    # Ensure temporary directory exists
    os.makedirs("._tmp", exist_ok=True)

    # Generate arguments list
    arguments_list = create_arguments_list()

    # Process each branch
    for branch in branches:
        checkout_and_pull_branch(repo, branch)
        if not os.path.exists(f"._tmp/{branch}/createLWE.exe"):
            build_cadet()
            shutil.copy("out/install/aRELEASE/bin/createLWE.exe", f"._tmp/{branch}/createLWE.exe")
        os.makedirs(f"._tmp/{branch}", exist_ok=True)
        for argument in arguments_list:
            run_create_lwe(branch, argument)

    # Compare LWE simulation results across branches
    compare_across_branches(arguments_list, branches)


if __name__ == '__main__':
    main()
