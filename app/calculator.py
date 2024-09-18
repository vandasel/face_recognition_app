def calculate_dict(TP,TN,FP,FN):
    return {
            "Accuracy" : (TP+TN)/(TP+TN+FP+FN),
            "Precision" : TP/(TP+FP) if TP+FP!=0 else 0,
            "Recall" : TP/(TP+FN) if TP+FN!=0 else 0,
            "Specifity" : TN/(TN+FP) if TN+FP!=0 else 0
        }

def get_best(dictt:dict):
    maks = 0
    best = ""
    for threshold,metrics in dictt.items():
        result = 0
        for metric in metrics.values():
            result+=metric
        if result>maks:
            maks = result
            best = threshold
    return best