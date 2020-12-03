#MET-IIB-4: Manufacturing Systems Engineering
#This is an exemple for saving your inspection results
#step1: import from save_results import ResultsSave
#step2: define ResultsSave('groupx_vision_result.csv','groupx_plc_result.csv')
#step3: insert 'insert_vision' and 'insert_plc' into your code
#please email me zl461@cam.ac.uk if you find any errors 

from save_results import ResultsSave

my_results=ResultsSave('group5_vision_result.csv','group5_plc_result.csv')


shuttle_list=['straight','curved']
i=0
while i<len(shuttle_list):
    
    #save vision system results
    for j in range(1,13):
        my_results.insert_vision(i,j,shuttle_list[i],'missing')


    #save plc message results
    my_results.insert_plc(i,0) #suppose '0' is 'pass'   
    #or 
    my_results.insert_plc(i,['0001']) #suppose '0001' is position '1'    

    i=i+1 

       