class DataLoader:
    def __init__(self, x, y, batch_size=16):
        self.x = x
        self.y = y
        self.batch_size = batch_size

        self._randomize()

        self.num_points = self.x.shape[0]

    def next_batch(self):
        '''
            Yields a batch (x, y) of size `batch_size`
        '''
        
        # Calculate number of batches.
        n_batches = int(np.ceil(self.num_points/self.batch_size))
        
        for b in range(n_batches):
            # Get start index of batch.
            start = b * self.batch_size
            
            # Get end index of batch.
            end = start + self.batch_size
            # Set limit to the ending index.
            end = min(end, self.num_points)
            
            # yielding will return a generator.
            yield (self.X[start: end], self.Y[start: end])

    def _randomize(self):
        '''
            Randomize the dataset.
        '''
        indices = np.arange(self.num_points)
        np.random.shuffle(indices)
        self.x = x[indices]
        self.y = y[indices]