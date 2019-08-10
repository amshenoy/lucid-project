############## USAGE ##############
from lucid_project.analysis import *
##frames = ["109883583_3_0","109883583_2_0","109883583_1_0","109883583_0_0","713576296_0_0"]
particles = {"109883583_3_0":"32","109883583_2_0":"32","713576296_0_0":"32", "1393200002_16_0":"6", "1686524462_0_0":"122"}
for frame_name, index in particles.items():
    frame = XYC.read("./sample_data/xyc_files/"+frame_name+".txt", "frame") ## Frame object
    
    frame.set_energy(True)
    ##frame.image().save("test.png")
    frame.plot().save("./sample_data/output/energy/frame/"+frame_name+".png")
    frame.cluster().plot().save("./sample_data/output/energy/clustering/"+frame_name+".png")
    clusters = frame.cluster().list
    ##print(len(clusters))
    cluster = clusters[int(index)]
    cluster.plot().save("./sample_data/output/energy/particle/"+frame_name+"_"+index+".png")


##    frame.set_energy(False)
##    ##frame.image().save("test.png")
##    frame.plot().save("./sample_data/output/tot/frame/"+frame_name+".png")
##    frame.cluster().plot().save("./sample_data/output/tot/clustering/"+frame_name+".png")
##    clusters = frame.cluster().list
##    cluster = clusters[int(index)]
##    cluster.plot().save("./sample_data/tot/particle/"+frame_name+"_"+index+".png")

    print(frame_name)
