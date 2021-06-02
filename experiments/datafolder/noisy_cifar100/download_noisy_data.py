import gdown

url = 'https://drive.google.com/uc?id=1FYgoksNhDr_2WwQ7L0nVgZ_d9MRpvTh6'
output = 'PointSegDAdataset.rar'
gdown.download(url, output, quiet=False)