import subprocess
import os 
import zipfile 
import shutil 
from tqdm import tqdm 



def download(genome_ids, path:str=None):

    archive_path = os.path.join(path, 'ncbi_dataset.zip')

    def extract(genome_id:str):
        # https://stackoverflow.com/questions/4917284/extract-files-from-zip-without-keeping-the-structure-using-python-zipfile
        archive = zipfile.ZipFile(archive_path)
        for member in archive.namelist():
            if member.startswith(f'ncbi_dataset/data/{genome_id}'):
                source = archive.open(member)
                # NOTE: Why does wb not result in another zipped file being created?
                with open(os.path.join(path, f'{genome_id}.fn'), 'wb') as target:
                    shutil.copyfileobj(source, target)

    for genome_id in tqdm(genome_ids, desc='download_genomes'):
        if not os.path.exists(os.path.join(path, f'{genome_id}.fna')):
            # Make sure to add a .1 back to the genome accession (removed while removing duplicates).
            cmd = f'datasets download genome accession {genome_id} --filename {archive_path} --include genome --no-progressbar'
            subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            extract(genome_id)
            os.remove(archive_path)
