import numpy as np

def compute_ece(confidences, labels, n_bins=10):
    """
    Compute Expected Calibration Error using quantile binning.
    
    Args:
        confidences: Array of prediction confidences/probabilities (floats between 0 and 1).
        labels: Array of true binary labels (0 or 1).
        n_bins: Number of bins to use (default=10 for decile binning).
    
    Returns:
        ece: Expected Calibration Error.
        bin_accuracies: Array of accuracies in each bin.
        bin_confidences: Array of mean confidences in each bin.
        bin_counts: Array of sample counts in each bin.
    """
    # Ensure inputs are numpy arrays
    confidences = np.array(confidences)
    labels = np.array(labels)
    
    # Input validation
    assert np.all((confidences >= 0) & (confidences <= 1)), "Confidences must be between 0 and 1."
    assert set(np.unique(labels)).issubset({0, 1}), "Labels must be binary (0 or 1)."
    
    # Calculate quantile bin edges
    bin_edges = np.percentile(confidences, np.linspace(0, 100, n_bins + 1))
    bin_edges[-1] = np.nextafter(bin_edges[-1], bin_edges[-1] + 1)  # Ensure the last prediction is included
    
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    # Iterate through bins
    for i in range(n_bins):
        # Find samples in the current bin
        mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        bin_count = np.sum(mask)
        if bin_count > 0:
            # Calculate accuracy and mean confidence in bin
            bin_accuracy = np.mean(labels[mask])
            bin_confidence = np.mean(confidences[mask])
        else:
            # Handle empty bins
            bin_accuracy = 0
            bin_confidence = 0
        
        bin_accuracies.append(bin_accuracy)
        bin_confidences.append(bin_confidence)
        bin_counts.append(bin_count)
    
    # Convert to numpy arrays
    bin_accuracies = np.array(bin_accuracies)
    bin_confidences = np.array(bin_confidences)
    bin_counts = np.array(bin_counts)
    
    # Calculate ECE
    total_samples = np.sum(bin_counts)
    ece = np.sum(np.abs(bin_accuracies - bin_confidences) * (bin_counts / total_samples))
    
    return {
        'ece': ece,
        'bin_accuracies': bin_accuracies,
        'bin_confidences': bin_confidences,
        'bin_counts': bin_counts
    }


