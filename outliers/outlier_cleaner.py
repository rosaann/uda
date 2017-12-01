#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    data_arr = []
    for pred, age, nw in zip(predictions, ages, net_worths):       
        dis = abs(pred[0] - nw[0])
        ob = {'dis': dis, 'age':age, 'net' : nw}
#        print "pred ", pred, " age ", age, "nw " , nw, "dis", dis, "ob ", ob
        data_arr.append(ob)
#    print data_arr
    data_arr = sorted(data_arr,reverse=False, key=lambda k: k['dis']) 
#    print data_arr
#    print "len ", len(data_arr), " * 0.9", (len(data_arr) * 0.9)
    for ob in data_arr[:int(len(data_arr) * 0.9) ]:
        cleaned_data.append((ob['age'], ob['net'], ob['dis']))
    print "all ", len(data_arr), "clearned ", len(cleaned_data)
    return cleaned_data

