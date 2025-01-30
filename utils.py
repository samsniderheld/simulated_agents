def list_to_txt(input_list: list, filename: str) -> None:
    """
    Writes each element of the input list to a new line in the specified .txt file.

    Args:
        input_list (list): The list of elements to write to the file.
        filename (str): The name of the output .txt file.
    """
    with open(f"{filename}.txt", 'w') as file:
        for element in input_list:
            file.write(f"{element}\n")

# Example usage:
# list_to_txt(["apple", "banana", "cherry"], "fruits")