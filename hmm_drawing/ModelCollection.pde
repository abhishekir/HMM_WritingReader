class ModelCollection {
  
  HashMap<Character, HMM> myModels;
  
  ModelCollection() {
    myModels = new HashMap<Character, HMM>();
  }
  
  void appendData(char modelname, int move) {
    HMM model = myModels.get(modelname);
    if (model == null) {
      HMM newModel = new HMM(modelname);
      newModel.addData(move);
      myModels.put(modelname, newModel);
      for (HMM m : myModels.values()) {
        m.reset_beliefs(); // Reset log probabilities and hidden state vector
      }
    } else {
      model.addData(move);
    }
  }
  
  void updateModel(char modelname) {
    HMM model = myModels.get(modelname);
    if (model == null) {
      System.err.println("Couldn't find model: " + modelname);
      return;
    }
    model.simpleTrain();
  }
  
  void updateBeliefs(int move) {
    for (HMM model : myModels.values()) {
      model.forward(move);
    }
  }
  
  void resetBeliefs() {
    for (HMM model : myModels.values()) {
      model.reset_beliefs();
    }
  }
  
  Character getPrediction() {
    double bestLogProb = Double.NEGATIVE_INFINITY;
    Character bestModel = '?';
    // Leave this debug information - the graders will find it useful as well
    System.err.println("Log likelihoods: ");
    for (Character m : myModels.keySet()) {
      System.err.print(m + ":");
      HMM model = myModels.get(m);
      double logProb = model.getLogProb();
      System.err.println(logProb);
      if (logProb > bestLogProb) {
        bestLogProb = logProb;
        bestModel = m;
      }
    }
    return bestModel;
  }
}
