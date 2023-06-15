# from transformers import BertTokenizer, AlbertForMaskedLM, FillMaskPipeline
# tokenizer = BertTokenizer.from_pretrained("albert-base-chinese-cluecorpussmall")
# model = AlbertForMaskedLM.from_pretrained("albert-base-chinese-cluecorpussmall")
# unmasker = FillMaskPipeline(model, tokenizer)
# res = unmasker("店名“小王黄焖鸡”很可能是一家[MASK][MASK]店")
# print('finished')


from transformers import BertTokenizer, AlbertModel
tokenizer = BertTokenizer.from_pretrained("albert-base-chinese-cluecorpussmall")
model = AlbertModel.from_pretrained("albert-base-chinese-cluecorpussmall")
text = "用你喜欢的任何文本替换我。"
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
print('finished')


# fine-tune