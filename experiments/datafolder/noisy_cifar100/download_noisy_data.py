import gdown

url = 'https://drive.google.com/uc?id=1V-Dwfl6RXVpmKB2lZFIehhJcE5_rYs8i'
output = 'data_dictionary.pkl'
gdown.download(url, output, quiet=False)