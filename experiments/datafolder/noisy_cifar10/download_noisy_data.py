import gdown

url = 'https://drive.google.com/uc?id=1clA-tCsfuCIbOTMHrHOHhj_TVJVzkDcy'
output = 'PointSegDAdataset.rar'
gdown.download(url, output, quiet=False)