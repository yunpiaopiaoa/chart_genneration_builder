from src.utils.dict_similarity import dict_similarity

# pred_dict = {
#     "header1": [1, 2, 3],
#     "header2": [4, 5, 6],
#     "header3": [7, 8, 9],
# }
# true_dict = {
#     "header1": [1, 2, 3],
#     "header2": [4, 5, 6],
#     "header3": [7, 8, 9],
# }

# pred_dict = {
#     "header1": [1, 2, 3],
#     "header2": [4, 5, 6],
#     "header4": [7, 8, 9],
#     "header3": [7, 8, -9],
# }
# true_dict = {
#     "header1": [1, 2, 3],
#     "header2": [4, 5, 6],
#     "header3": [7, 8, 9],
#     "header4": [7, 8, 9],
# }

# pred_dict = {
#     "header1": [1, 2, 3],
#     "header2": [4, 5, 6],
#     "header4": [7, 8, 9],
#     "header3": [7, 8, -9],
# }
# true_dict = {
#     "header5": [1, 2, 3],
#     "header6": [4, 5, 6],
#     "header7": [7, 8, 9],
#     "header8": [7, 8, 9],
# }

pred_dict = {
    "header1": [1, 2, 3],
    "header2": [4, 5, 6],
    "header4": [7, 8, 9],
    "header3": [7, 8, -9],
}
true_dict = {
    "header5": [10,11,12],
    "header6": [13,14,15],
    "header7": [16,17,18],
    "header8": [19,20,21],
    "header9": [22,23,24],
    "header10": [25,26,27],
}

score = dict_similarity(pred_dict, true_dict)
print(score)
