import os

def check_dataset(base_path):
    """check if dataset is correctly prepared"""
    ham_path = os.path.join(base_path, 'ham')
    spam_path = os.path.join(base_path, 'spam')
    
    # check if data path exists
    if not (os.path.exists(ham_path) and os.path.exists(spam_path)):
        print("Error: data dir not exist!")
        return False
    
    # count email number
    ham_count = len([f for f in os.listdir(ham_path) if f.endswith('.txt')])
    spam_count = len([f for f in os.listdir(spam_path) if f.endswith('.txt')])
    
    print(f"Normal Email Number: {ham_count}")
    print(f"Spam Email Number: {spam_count}")
    print(f"Total: {ham_count + spam_count}")
    
    # check if readable
    try:
        ham_file = os.path.join(ham_path, os.listdir(ham_path)[0])
        with open(ham_file, 'r', encoding='utf-8') as f:
            f.read()
            
        spam_file = os.path.join(spam_path, os.listdir(spam_path)[0])
        with open(spam_file, 'r', encoding='utf-8') as f:
            f.read()
            
        print("File readable, you are all set!")
        return True
    except Exception as e:
        print(f"Error: not readable\n{str(e)}")
        return False

# usage
dataset_path = '../../data/enron_data/enron1'  # path to enron1 data
check_dataset(dataset_path)