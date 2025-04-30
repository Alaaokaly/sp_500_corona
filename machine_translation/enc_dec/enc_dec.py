
import sys
import os 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from torch import nn 
import collections
from torch.nn import functional as F
from data_processing import data
import logging 

import time 
log_dir = 'machine_translation/log'
os.makedirs(log_dir, exist_ok= True)
log_file = os.path.join(log_dir, f"training_deu_eng.log")
logging.basicConfig(level=logging.INFO, 
                    format = '%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ])
logger = logging.getLogger(__name__)

class Trainer:
    def  __init__(self ,model, train_loader, val_loader = None, optimizer= None,
                  max_epochs= 10, grad_clip_val =0.0,  
                  device = 'cpu', num_gpus = 2): 
        self.model = model
        self.trainer_loader = train_loader 
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.gradient_clip_val  = grad_clip_val
        self.device = device
        self.num_gpus = num_gpus

        if self.num_gpus > 0:
           if torch.cuda.is_available():
               actual_gpus = min(torch.cuda.device_count(), self.num_gpus)
               if actual_gpus > 1:
                   self.model = torch.nn.DataParallel(self.model, device_ids=list(range(actual_gpus)))
                   self.device = 'cuda'
                   logger.info(f"Using {actual_gpus} GPUs for training")
               else:
                   self.device = 'cuda:0'
                   logger.info("Using 1 GPU for training")
               self.model = self.model.to(self.device)
           else:
               logger.warning("GPUs requested but CUDA is not available. Falling back to CPU.")
               self.device = 'cpu'
               self.num_gpus = 0
    
    def train_one_epoch(self, epoch = None):
        self.model.train()
        total_loss = 0
        logger.info(f"Starting epoch {epoch}/{self.max_epochs}")\
        
        batch_count = len(self.trainer_loader)

        for batch_idx, batch in enumerate(self.trainer_loader):
            src_array, tgt_array, src_valid_len, label_array = batch
            src_array = src_array.to(self.device)
            tgt_array = tgt_array.to(self.device)
            src_valid_len = src_valid_len.to(self.device)
            label_array = label_array.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(src_array, tgt_array)  # Use target array for decoder input
            loss = self.model.loss(outputs, label_array)# compute backward gradients 
            if self.gradient_clip_val > 0:
                # calculate the sum norm of params 
                #if norm >grad_clip_valu 
                # param_grid *= grad_clip_val/ norm
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.gradient_clip_val)
            self.optimizer.step() #update the new values for params using grad 
            total_loss+= loss.item()

            if batch_idx % 100 == 0:
               logger.info(f"Epoch {epoch}: {batch_idx}/{batch_count} batches processed. Current loss: {loss.item():.4f}")
    
    
        avg_loss = total_loss / len(self.trainer_loader)
        logger.info(f"Epoch {epoch} completed. Average training loss: {avg_loss:.4f}")
        return avg_loss
    def validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
             for x, y in self.val_loader:  # Unpack directly as x, y
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)
                loss = self.model.loss(outputs, y)
                total_loss += loss.item()
        val_loss = total_loss / len(self.val_loader)
        logger.info(f"Validation  Loss: {val_loss:.4f} ")
        return val_loss


    def fit (self):
        logger.info("=" * 50)
        logger.info(f"Starting training for {self.max_epochs} epochs")
        logger.info(f"Training on device: {self.device}")
        logger.info(f"Training set size: {len(self.trainer_loader.dataset)} examples")
        if self.val_loader:
            logger.info(f"Validation set size: {len(self.val_loader.dataset)} examples")
        logger.info("=" * 50)
        
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        for epoch in range(1, self.max_epochs+1):
            # Log epoch start
            logger.info(f"Epoch {epoch}/{self.max_epochs} started")
            
            # Train
            start_time = time.time()
            train_loss = self.train_one_epoch(epoch)
            train_losses.append(train_loss)
            epoch_time = time.time() - start_time
            
            # Validate if validation set exists
            if self.val_loader:
                val_loss = self.validate()
                val_losses.append(val_loss)
                
                # Log improvement in validation loss
                if val_loss < best_val_loss:
                    improvement = best_val_loss - val_loss
                    best_val_loss = val_loss
                    logger.info(f"Validation loss improved by {improvement:.4f}. New best: {best_val_loss:.4f}")
                
                logger.info(f"Epoch {epoch}/{self.max_epochs} completed in {epoch_time:.1f}s - "
                         f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                logger.info(f"Epoch {epoch}/{self.max_epochs} completed in {epoch_time:.1f}s - "
                         f"Train Loss: {train_loss:.4f}")
        
        # Log training summary
        logger.info("=" * 50)
        logger.info(f"Training completed after {self.max_epochs} epochs")
        if self.val_loader:
            logger.info(f"Best validation loss: {best_val_loss:.4f}")
        logger.info("=" * 50)
    


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward( self, *args):
        
        raise NotImplementedError
    
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

    def init_state(self, enc_all_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError
    
# encoders and decoder should has the same number of :
           #layers
           #hidden units 
class EncoderDecoder(nn.Module):
    def __init__(self, encoder:Encoder, decoder:Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder 
    def forward (self, enc_x, dec_x, *args):
        enc_all_outputs = self.encoder(enc_x, *args)
        dec_state = self.decoder.init_state(enc_all_outputs,*args)
        return self.decoder(dec_x, dec_state)[0]
    
# the encoded C should be concated with
#    the decoders input at all tims steps 

def init_seq2seq(module):
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight)
    if type(module) == nn.GRU :
        for param in module._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(module._parameters[param])
class Seq2SeqEncoder(Encoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,dropout=0.0):

        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens,num_layers, dropout=dropout)
        self.apply(init_seq2seq)
        
    def forward(self, x, *args):
        embs = self.embedding(x.t().long())
        outputs, state  = self.rnn(embs)
        return outputs, state
class  Seq2SeqDecoder(Decoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, 
                 num_layers, dropout = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size+ num_hiddens, num_hiddens
                                        , num_layers, dropout= dropout)
        self.dense = nn.LazyLinear(vocab_size)
        self.apply(init_seq2seq)

    def init_state(self, enc_all_outputs, *args):
        return enc_all_outputs

    def forward(self, x, state):
        embs = self.embedding (x.t().long())
        enc_output, hidden_state = state
        context = enc_output[-1]
        context = context.repeat(embs.shape[0],1,1)
        embs_and_context = torch.cat((embs,context), -1)
        outputs, hidden_state = self.rnn(embs_and_context, hidden_state)
        outputs = self.dense(outputs).swapaxes(0,1)
        return outputs, [enc_output, hidden_state]


class Seq2Seq (EncoderDecoder):
    def __init__(self, encoder, decoder, tgt_pad, lr):
        super().__init__(encoder, decoder)
        self.tgt_pad = tgt_pad
        self.lr = lr

    def configure_optimization(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    def loss (self, y_hat, y):
     
    
        if y_hat.ndim == 3:  
  
           y_hat = y_hat.reshape(-1, y_hat.shape[-1])
           y = y.reshape(-1)
    

        l = F.cross_entropy(y_hat, y.long(), reduction='none')
        
     
        mask = (y.reshape(-1) != self.tgt_pad).float()
        return (l * mask).sum() / mask.sum()
    

data = data.MTDeuEng(batch_size=128)

train_loader = data.get_dataloader(train = True)
val_loader = data.get_dataloader(train=False)

embed_size, num_hiddens, num_layers, dropout = 256, 256, 2, 0.2

encoder = Seq2SeqEncoder(
    len(data.src_vocab), embed_size, num_hiddens, num_layers, dropout)

decoder = Seq2SeqDecoder(
    len(data.tgt_vocab), embed_size, num_hiddens, num_layers, dropout)

model = Seq2Seq(encoder, decoder, tgt_pad=data.tgt_vocab['<pad>'],
                lr=0.005)

logger.info("=" * 70)
logger.info("TRAINING CONFIGURATION")
logger.info(f"Source vocabulary size: {len(data.src_vocab)}")
logger.info(f"Target vocabulary size: {len(data.tgt_vocab)}")
logger.info(f"Embedding size: {embed_size}, Hidden size: {num_hiddens}")
logger.info(f"Layers: {num_layers}, Dropout: {dropout}")
logger.info(f"Batch size: {data.batch_size}")
logger.info(f"Learning rate: {model.lr}")
logger.info("=" * 70)
opt = model.configure_optimization()
trainer = Trainer(model,train_loader=train_loader, max_epochs=30, 
                  grad_clip_val=1, num_gpus=2, optimizer=opt)


# 10. Make sure to log when starting the actual training
logger.info("Starting model training...")

trainer.fit()
logger.info("Training completed!")