import os
from tkinter import Tk, filedialog, simpledialog

def select_files():
    """Opens the dialogue to select specific files for string append."""
    root = Tk()
    root.withdraw()
    files = filedialog.askopenfilenames(title="Select files to rename")
    return root.tk.splitlist(files)

def get_string_to_append():
    """Prompt for a string to append to the filenames."""
    root = Tk()
    root.withdraw()
    user_input = simpledialog.askstring("Input String", "Enter the string to append: FORMAT: SITE_PARTICIPANTCODE_TESTSEQUENCE")
    return user_input

def rename_files(files, append_str):
    """Rename the selected files by appending the user input string at the start of filename"""
    for file in files:
        dir_name = os.path.dirname(file)
        base_name = os.path.basename(file)
        name, ext = os.path.splitext(base_name)
        new_name = f"{append_str}{name}{ext}" # adds the custom string at the start of the file, but can change to the end if needed, by moving {append_str} after {name}
        new_path = os.path.join(dir_name, new_name)
        os.rename(file, new_path)
        #print(f"Renamed '{file}' to '{new_path}'")

def main():
    files = select_files()
    if not files:
        print("No files selected.") # TODO: make this actually print as a log message 
        return

    append_str = get_string_to_append()
    if append_str is None:
        print("No string provided.") # TODO: make this actually print as a log message 
        return

    rename_files(files, append_str)
    print("Files renamed successfully.") # TODO: make this actually print as a log message 

if __name__ == "__main__":
    main()