import argparse

def get_data_script_arg_parser():
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
    "index_path", help="The path to the data index created with index_data.py.")

  parser.add_argument(
    "--data_path",
    nargs="?",
    default=None,
    help="The path to the data set. If None or empty, the parent of index_path is used.")

  return parser
