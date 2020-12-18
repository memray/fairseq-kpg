# -*- coding: utf-8 -*-
"""
Python File Template 
"""
import json
import os

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"

if __name__ == '__main__':
    vocab_path = '/zfs1/hdaqing/rum20/kp/fairseq-kpg/fairseq_cli/data-bin/hf_vocab/special_vocab'

    # with open(vocab_path, 'r') as vocab_file:
    #     for line in vocab_file:
    #         # print(line.strip().lower())
    #         print(line)

    # path = '/zfs1/pbrusilovsky/rum20/kp/OpenNMT-kpg/output/keyphrase/meng17-one2one/meng17-one2one-kp20k-v3/meng17-one2one-fullbeam/meng17-one2one-beam200-maxlen6/pred/kpgen-meng17-kp20k-one2one-transformer-L6H8-BS4096-LR0.05-L6-H8-Dim512-Emb512-Dropout0.1-Copyfalse-Covfalse_step_5000/inspec.pred'
    #
    # with open(path, 'r') as json_file:
    #     for lid, line in enumerate(json_file):
    #         print(lid)
    #         try:
    #             ex_dict = json.loads(line)
    #         except:
    #             print(line)
    #

    json_str = '{"src": [25, 11183, 27, 710, 11183, 1337, 164, 36, 7, 6, 2816, 1390, 196, 765, 14, 71, 20, 86, 10, 86, 11183, 1391, 60, 74, 152, 21, 196, 5084, 154, 16, 26985, 17, 7, 3319, 9, 12, 4705, 4760, 5084, 154, 16, 4174, 17, 1600, 97, 1688, 26985, 1456, 1259, 4813, 425, 44, 14, 632, 15, 11183, 168, 12, 1487, 140, 1750, 278, 15, 1989, 168, 469, 81, 161, 2206, 7019, 7, 139, 9, 15, 339, 1391, 20, 60, 74, 3306, 21, 4760, 5084, 154, 16, 4174, 17, 9, 1949, 47, 10, 10901, 1117, 134, 32, 6460, 12, 16787, 280, 21, 26985, 7, 13, 18, 37, 9, 23, 142, 27, 710, 11183, 1337, 164, 36, 12, 2801, 21, 62, 151, 7, 21, 6, 520, 8, 6, 710, 164, 36, 9, 6, 654, 1750, 8, 4174, 145, 1391, 38, 32, 708, 3746, 12, 1688, 11183, 312, 13, 27, 163, 282, 7, 28, 11, 3251, 9, 6, 1117, 456, 21, 1714, 11183, 1391, 407, 32, 426, 638, 7, 6, 164, 36, 76, 344, 6, 36, 3141, 12, 1135, 6, 1756, 8, 11175, 10, 2294, 15, 3791, 17050, 54, 13845, 1750, 7, 6, 654, 1750, 134, 32, 638, 10, 20431, 12, 2076, 6, 317, 8, 161, 2206, 7019, 9, 343, 1001, 6, 149, 224, 10, 2227, 6184, 7, 572, 49, 2661, 130, 214, 261, 60, 0, 6, 258, 56, 8, 6, 11183, 164, 36, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], "src_raw": ["i", "wap", "an", "intelligent", "wap", "site", "management", "system", ".", "the", "popularity", "regarding", "wireless", "communications", "is", "such", "that", "more", "and", "more", "wap", "sites", "have", "been", "developed", "with", "wireless", "markup", "language", "(", "wml", ")", ".", "meanwhile", ",", "to", "translate", "hypertext", "markup", "language", "(", "html", ")", "pages", "into", "proper", "wml", "ones", "becomes", "imperative", "since", "it", "is", "difficult", "for", "wap", "users", "to", "read", "most", "contents", "designed", "for", "pc", "users", "via", "their", "mobile", "phone", "screens", ".", "however", ",", "for", "those", "sites", "that", "have", "been", "maintained", "with", "hypertext", "markup", "language", "(", "html", ")", ",", "considerable", "time", "and", "manpower", "costs", "will", "be", "incurred", "to", "rebuild", "them", "with", "wml", ".", "in", "this", "paper", ",", "we", "propose", "an", "intelligent", "wap", "site", "management", "system", "to", "cope", "with", "these", "problems", ".", "with", "the", "help", "of", "the", "intelligent", "management", "system", ",", "the", "original", "contents", "of", "html", "web", "sites", "can", "be", "automatically", "translated", "to", "proper", "wap", "content", "in", "an", "efficient", "way", ".", "as", "a", "consequence", ",", "the", "costs", "associated", "with", "maintaining", "wap", "sites", "could", "be", "significantly", "reduced", ".", "the", "management", "system", "also", "allows", "the", "system", "manager", "to", "define", "the", "relevance", "of", "numerals", "and", "keywords", "for", "removing", "unimportant", "or", "meaningless", "contents", ".", "the", "original", "contents", "will", "be", "reduced", "and", "reorganized", "to", "fit", "the", "size", "of", "mobile", "phone", "screens", ",", "thus", "reducing", "the", "communication", "cost", "and", "enhancing", "readability", ".", "numerical", "results", "gained", "through", "various", "experiments", "have", "evinced", "the", "effective", "performance", "of", "the", "wap", "management", "system"], "gold_sent": [["wireless", "mobile", "internet"], ["communication", "cost"], ["wireless", "mobile", "internet"], ["wireless", "mobile", "internet"], ["mobile", "phone"], ["wireless", "mobile", "internet"], ["html", "pages"], ["communication", "cost"]], "gold_score": 0, "word_aligns": null, "attns": [], "copied_flags": [], "unique_pred_num": 0, "dup_pred_num": 0, "beam_num": 0, "beamstep_num": 0, "pred_sents": [], "pred_scores": [], "preds": [], "ori_pred_sents": null, "ori_pred_scores": null, "ori_preds": null, "topseq_pred_sents": null, "topseq_pred_scores": null, "topseq_preds": null, "dup_pred_tuples": null}'
    ex_dict = json.loads(json_str)
    print(ex_dict)
