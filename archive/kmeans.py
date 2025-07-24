import numpy as np
from util import DataExtractor

##wurde im runpod4-Branch verwendet um beste Bounding Boxes zu identifizieren

class KMeans:
    def __init__(self, sample_size: int = 20000, seed: int = None):
        """
        sample_size: how many annotations to sample for clustering
        seed:        random seed for reproducibility
        """
        # Extract and normalize your annotations DataFrame
        de = DataExtractor()
        self.anns_df = de.normalizedData()  
        
        # Optionally subsample
        if sample_size < len(self.anns_df):
            df_sample = self.anns_df.iloc[:sample_size]
        else:
            df_sample = self.anns_df
        
        # Build (w,h) array
        self.boxes = df_sample[['w', 'h']].to_numpy()  # shape (M,2)
        
        # Store random seed
        self.seed = seed
        
        # Placeholders for results
        self.anchors_ = None
        self.avg_iou_ = None

    def iou(self, box: np.ndarray, clusters: np.ndarray) -> np.ndarray:
        """
        Compute the IoU between one box (w,h) and each cluster centroid.
        Returns an array of IoUs of shape (num_clusters,).
        """
        bw, bh = box
        cw, ch = clusters[:, 0], clusters[:, 1]
        inter_w = np.minimum(bw, cw)
        inter_h = np.minimum(bh, ch)
        inter_area = inter_w * inter_h
        box_area = bw * bh
        cluster_area = cw * ch
        union_area = box_area + cluster_area - inter_area
        return inter_area / union_area

    def kmeans_iou(self, k: int, iterations: int = 100) -> tuple[np.ndarray, float]:
        """
        Run k-means clustering (with IoU-based distance) on self.boxes.
        
        Args:
          k:           number of clusters (anchors) to find
          iterations:  maximum number of k-means iterations
        
        Returns:
          anchors: numpy array of shape (k,2) sorted by area
          avg_iou: average IoU between each box and its nearest anchor
        """
        # Set seed for reproducibility, if provided
        if self.seed is not None:
            np.random.seed(self.seed)

        # 1) Initialize centroids by sampling k boxes
        indices = np.random.choice(len(self.boxes), k, replace=False)
        centroids = self.boxes[indices].copy()

        for _ in range(iterations):
            # 2) Compute "distance" = 1 - IoU(box, each centroid)
            distances = 1 - np.stack([self.iou(b, centroids) for b in self.boxes])
            
            # 3) Assign each box to its nearest centroid
            assignments = np.argmin(distances, axis=1)
            
            # 4) Recompute centroids as the mean of assigned boxes
            new_centroids = np.array([
                self.boxes[assignments == i].mean(axis=0)
                if np.any(assignments == i) else centroids[i]
                for i in range(k)
            ])
            
            # 5) Check for convergence
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids

        # 6) Sort the resulting anchors by area (width*height)
        areas = centroids[:, 0] * centroids[:, 1]
        order = np.argsort(areas)
        anchors = centroids[order]

        # 7) Compute final average Io
        final_dist = 1 - np.stack([self.iou(b, anchors) for b in self.boxes])
        avg_iou = float(np.mean(1 - np.min(final_dist, axis=1)))

        # Store results
        self.anchors_ = anchors
        self.avg_iou_ = avg_iou

        return anchors, avg_iou

