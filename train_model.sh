#!/bin/bash
# Quick script to train LayoutLM model end-to-end

echo "======================================================================"
echo "LAYOUTLM TRAINING PIPELINE"
echo "======================================================================"

# Activate virtual environment
source venv/bin/activate

# Step 1: Prepare data
echo ""
echo "Step 1/3: Preparing training data..."
python src/ml_training/prepare_data.py
if [ $? -ne 0 ]; then
    echo "❌ Data preparation failed!"
    exit 1
fi

# Step 2: Train model
echo ""
echo "Step 2/3: Training LayoutLM model..."
python src/ml_training/train_layoutlm.py
if [ $? -ne 0 ]; then
    echo "❌ Training failed!"
    exit 1
fi

# Step 3: Evaluate
echo ""
echo "Step 3/3: Evaluating model..."
python src/ml_training/evaluate.py --num 50
if [ $? -ne 0 ]; then
    echo "❌ Evaluation failed!"
    exit 1
fi

echo ""
echo "======================================================================"
echo "✓ TRAINING COMPLETE!"
echo "======================================================================"
echo ""
echo "Model saved to: models/layoutlm_invoice/final_model/"
echo ""
echo "Next steps:"
echo "  1. Test on single invoice:"
echo "     python src/ml_training/predict.py --image sample_documents/invoice_dataset/image/001.png"
echo ""
echo "  2. Integrate into your pipeline"
echo ""

