for i in /home/azureuser/hackathon_data/hackathon_data/*
do 
#   echo "$i"
  rosbag reindex "$i"/*.bag.active
  rosbag fix "$i"/*.bag.active "$i"/data.bag
done