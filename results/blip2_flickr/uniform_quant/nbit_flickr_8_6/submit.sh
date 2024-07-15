#!/bin/bash
#SBATCH --array=1-336%80
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --requeue
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --gres=gpu:8
#SBATCH --exclude=brigid16,brigid17,brigid18,brigid19,cbcb00,cbcb01,cbcb02,cbcb03,cbcb04,cbcb05,cbcb06,cbcb07,cbcb08,cbcb09,cbcb10,cbcb11,cbcb12,cbcb13,cbcb14,cbcb15,cbcb16,cbcb17,cbcb18,cbcb19,cbcb20,cbcb21,cbcb22,cbcb23,cbcb24,cbcb25,cbcb28,cbcb29,clip00,clip01,clip02,clip03,clip04,clip07,clip08,clip09,clip10,cml00,cml01,cml02,cml03,cml04,cml05,cml06,cml07,cml08,cml09,cml10,cml11,cml12,cml13,cml14,cml15,cml16,cml31,cml32,cmlcpu00,cmlcpu01,cmlcpu02,cmlcpu03,cmlcpu04,cmlcpu06,cmlcpu07,janus02,janus03,janus04,legacy00,legacy01,legacy02,legacy03,legacy04,legacy05,legacy06,legacy07,legacy08,legacy09,legacy10,legacy11,legacy13,legacy14,legacy15,legacy16,legacy17,legacy18,legacy19,legacy20,legacy21,legacy22,legacy23,legacy24,legacy25,legacy26,legacy27,legacy28,legacy30,legacygpu00,legacygpu01,legacygpu02,legacygpu03,legacygpu04,legacygpu05,legacygpu06,legacygpu07,mbrc00,mbrc01,oasis00,oasis01,oasis02,oasis03,oasis04,oasis05,oasis06,oasis07,oasis08,oasis09,oasis10,oasis11,oasis12,oasis13,oasis14,oasis15,oasis16,oasis17,oasis18,oasis19,oasis20,oasis21,oasis22,oasis23,oasis24,oasis25,oasis26,oasis27,oasis28,oasis29,oasis30,oasis31,oasis32,oasis33,oasis34,oasis35,oasis36,oasis37,oasis38,oasis39,oasis40,quics00,tron62,tron63,tron64,tron65,tron66,tron67,tron68,tron69,twist00,twist01,twist02,twist03,twist04,twist05,vulcan00,vulcan01,vulcan02,vulcan03,vulcan04,vulcan05,vulcan06,vulcan07,vulcan08,vulcan09,vulcan10,vulcan11,vulcan12,vulcan13,vulcan14,vulcan15,vulcan16,vulcan17,vulcan18,vulcan19,vulcan20,vulcan21,vulcan22,vulcan23,vulcan25,vulcan26,vulcan27,vulcan28

cd /vulcanscratch/elau1/lavis_clone
source ~/ViT/bin/activate
srun --output=$(head -n $SLURM_ARRAY_TASK_ID /vulcanscratch/elau1/lavis_clone/slurm_files/nbit_flickr/log.txt | tail -n 1)  $(head -n $(expr 1 \* $SLURM_ARRAY_TASK_ID - 0) /vulcanscratch/elau1/lavis_clone/slurm_files/nbit_flickr/now.txt | tail -n 1)

