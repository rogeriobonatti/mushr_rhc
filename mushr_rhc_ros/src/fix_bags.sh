for i in /home/azureuser/hackathon_data_premium/hackathon_data_2p5_withpartialnoise0/*
do 
#   echo "$i"
  rosbag reindex "$i"/*.bag.active
  rosbag fix "$i"/*.bag.active "$i"/data.bag
done