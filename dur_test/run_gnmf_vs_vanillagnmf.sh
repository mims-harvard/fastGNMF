#!/bin/bash
#SBATCH -n 1 # Number of cores requested
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t 120 # Runtime in minutes
#SBATCH -p serial_requeue # Partition to submit to
#SBATCH --mem=1G # Memory in GB
#SBATCH -o gnmf_slurm.out # Standard out goes to this file
#SBATCH -e gnmf_slurm.err # Standard err goes to this file

n_list=(100 500 1000 5000 10000 30000)
m_list=(1000 1000 5000 10000 50000 50000)
n_list=(100)
m_list=(50)
p=8
k=20

for i in ${!n_list[*]}; do
  n=${n_list[i]}
  m=${m_list[i]}
  printf "#!/bin/bash\npython dur_test/run_gnmf.py -n $n -m $m -p $p -k $k" > dur_test/run_gnmf_dum.sh
  sbatch dur_test/run_gnmf_dum.sh -p serial_requeue -t 0-02:00 --mem=5G -o gnmf_slurm_${n}_${m}.out -e gnmf_slurm_${n}_${m}.err
  sleep 1
  # echo "python dur_test/run_gnmf.py -n $n -m $m -p $p -k $k --vanilla" > dur_test/run_gnmf_dum_van.sh
  # sbatch dur_test/run_gnmf_dum_van.sh -p serial_requeue -t 120 --mem=5G -o gnmf_vanilla_${n}_${m}_slurm.out -e gnmf_vanilla_${n}_${m}_slurm.err
done
#
# source activate ${env_name}
#
# for letter in {W..Z}
# do
#   fastq="/n/stat115/2020/HW2/raw_data/run${letter}.fastq.gz"
#
#   STAR --genomeDir /n/stat115/2020/HW2/star_hg38_index \
#   --readFilesIn ${fastq} \
#   --runThreadN 1 --outFileNamePrefix star_mapping_${letter}.out \
#   --outSAMtype BAM SortedByCoordinate \
#   --readFilesCommand zcat
#
#   bam="star_mapping_${letter}.outAligned.sortedByCoord.out.bam"
#   samtools index ${bam}
#   ~/.conda/envs/${env_name}/bin/tin.py -i ${bam} -r ${bed}
# done
