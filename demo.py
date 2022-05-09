import pdb
import transtab

allset, trainset, valset, testset, cat_cols, num_cols, bin_cols \
     = transtab.load_data(['credit-approval', 'credit-g'])
model, collate_fn = transtab.build_contrastive_learner(cat_cols, num_cols, bin_cols)
transtab.train(model, allset, collate_fn=collate_fn)


# allset, trainset, valset, testset, cat_cols, num_cols, bin_cols \
#      = transtab.load_data('credit-g')
# model = transtab.build_classifier(
#     cat_cols, num_cols, bin_cols,
# )

# allset, trainset, valset, testset, cat_cols, num_cols, bin_cols \
#      = transtab.load_data('credit-approval')
# model.update({'cat':cat_cols,'num':num_cols, 'bin':bin_cols})

# model, collate_fn = transtab.build_contrastive_learner(checkpoint='./ckpt')
# transtab.train(model, [trainset,valset,testset], collate_fn=collate_fn, **training_arguments)




# model, collate_fn = transtab.build_contrastive_learner(
#     cat_cols, num_cols, bin_cols,
# )



training_arguments = {
    'num_epoch':5,
    'lr':1e-4,
    'weight_decay':0,
    'batch_size':64,
    'patience':10,
    'warmup_ratio':0.1,
    'output_dir':'./ckpt',
    'eval_metric':'auc',
    'num_workers':4, # for small dataset (<10000), recommend to set it 0
}



# transtab.train(model, trainset, valset, **training_arguments)

# ypred = transtab.predict(model, valset[0])
# transtab.evaluate(ypred, valset[1], 'auc')

# model = transtab.build_classifier(
#     checkpoint = './ckpt'
# )
# model.save('./ckpt')

# extractor = transtab.build_extractor(checkpoint='./ckpt')
# extractor.save('./ckpt')
# extractor.load('./ckpt/extractor')

pdb.set_trace()
pass
