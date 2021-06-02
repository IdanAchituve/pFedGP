import gdown

url = 'https://drive.google.com/uc?id=1tz_lMUHojjXvdHiVaaBEw1fwItY1BrEc'
output = 'data_dictionary.pkl'
gdown.download(url, output, quiet=False)