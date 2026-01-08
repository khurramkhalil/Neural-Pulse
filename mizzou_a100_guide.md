# How to Use Mizzou A100 GPUs on Nautilus

This guide explains how to configure your Kubernetes jobs to specifically target the reserved A100 GPUs available to the Mizzou group.

## 1. Namespace
You must deploy your jobs in the Mizzou namespace:
```yaml
metadata:
  namespace: gp-engine-mizzou-dcps
```

## 2. Resource Request
Instead of the generic `nvidia.com/gpu`, you should request the specific A100 resource type if available, or use the generic type with affinity.
**Recommended Pattern (from existing jobs):**
```yaml
resources:
  requests:
    nvidia.com/a100: 1  # Request 1 A100 GPU
    memory: "32Gi"
    cpu: "8"
  limits:
    nvidia.com/a100: 1
    memory: "32Gi"
    cpu: "8"
```

## 3. Node Affinity (Targeting A100s)
Ensure your job lands on a node with A100 GPUs by adding a node affinity rule:
```yaml
spec:
  affinity:
    nodeAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
        - matchExpressions:
          - key: nvidia.com/gpu.product
            operator: In
            values:
            - NVIDIA-A100-SXM4-80GB
            - NVIDIA-A100-80GB-PCIe
```

## 4. Tolerations (Allocating Reserved Nodes)
Mizzou nodes are often tainted to prevent general usage. You must add a specific toleration to bypass this taint and use the reserved hardware:
```yaml
spec:
  tolerations:
  - key: "nautilus.io/reservation"
    operator: "Equal"
    value: "mizzou"
    effect: "NoSchedule"
```

## 5. Full Example Job YAML
Combine all the above into your job manifest:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: my-mizzou-a100-job
  namespace: gp-engine-mizzou-dcps
spec:
  template:
    spec:
      restartPolicy: Never
      
      # 1. Toleration for Mizzou Reservation
      tolerations:
      - key: "nautilus.io/reservation"
        operator: "Equal"
        value: "mizzou"
        effect: "NoSchedule"
        
      # 2. Affinity for A100 Hardware
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: nvidia.com/gpu.product
                operator: In
                values:
                - NVIDIA-A100-SXM4-80GB
                - NVIDIA-A100-80GB-PCIe

      containers:
      - name: training-container
        image: khurramkhalil/gpa-emergency:latest
        resources:
          # 3. Requesting the Resource
          requests:
            nvidia.com/a100: 1
            memory: "16Gi"
            cpu: "4"
          limits:
            nvidia.com/a100: 1
            memory: "16Gi"
            cpu: "4"
        command: ["/bin/bash", "-c"]
        args: ["python examples/train_lra.py ..."]
```

