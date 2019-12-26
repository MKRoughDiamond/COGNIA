class CogModule:
    def __init__(self,model,model_parameter,**kwargs):
        raise NotImplementedError

    def set_gpu(self, num_gpu=1, device_id=None,**kwargs):
        raise NotImplementedError

    def set_cpu(self,**kwargs):   
        raise NotImplementedError

    def set_optimizers(self,
                optimizer=None,
                loss_fn=None,
                LEARNING_RATE=None,
                scheduler=None,
                **kwargs):
        raise NotImplementedError

    def set_dataloader(self,
                PATH=None,
                BATCH_SIZE=None,
                IS_TRAIN=True,
                **kwargs):
        raise NotImplementedError

    def fit(self,dataloader,NUM_EPOCHES,**kwargs):
        raise NotImplementedError

    def predict(self,x,**kwargs):
        raise NotImplementedError

    def get_loss(self,x,y,**kwargs):
        raise NotImplementedError

    def save_model(self,path,**kwargs):
        raise NotImplementedError

    def load_model(self,path,**kwargs):
        raise NotImplementedError
