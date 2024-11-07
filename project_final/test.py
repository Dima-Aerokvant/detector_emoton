def predominant_emotion( list):
    result = {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 0,'positive': 0,'other' : 0}
    for elem in list:
        for key in elem:
            if result[key] == 0:
                result[key] = elem[key]
            else:
                result[key] = (result[key] + elem[key])/2
    result['positive'] = (result['positive']+ result['happy'])/2
    result.pop('happy')
    print("result:",result)
    return result

print(predominant_emotion([{'other':0.9695},{'other':0}]))
