import gdown

url = 'https://drive.google.com/uc?id=1tz_lMUHojjXvdHiVaaBEw1fwItY1BrEc'
output = 'PointSegDAdataset.rar'
gdown.download(url, output, quiet=False)