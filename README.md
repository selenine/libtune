# libtune

libtune is a work in progress derivative-free/zeroth-order optimization library I'm writing to optimize [Episteme](https://github.com/aletheiaaaaa/episteme), a high performance chess engine. Instead of training Episteme's neural network via supervised learning on a large amount of scored positions, I would like to train that neural network end-to-end from games themselves (even though a direct win/draw/loss score is nondifferentiable). This approach has shown promise on a small scale in top engines such as Stockfish, and I would like to push beyond the limits of existing frameworks to train, or at least fine-tune, the whole neural network.

So far, I've built all the optimizers I would want to try out, including SGD, Adam and Muon, and I'm left with implementing the gradient estimators for them. After that, the library should be usable by anyone (though some intentional design decisions mean it's primary users will probably be other engine developers), though I plan to add learning rate schedulers too.

I hope you enjoy!

