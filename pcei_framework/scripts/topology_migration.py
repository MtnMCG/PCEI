
import torch
import torch.nn as nn
import argparse
from pathlib import Path
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pcei import PCEINexus
from src.modules.plasticity import PlasticLayer

def migrate_weights(src_model: nn.Module, dst_model: nn.Module):
    """
    Surgically migrates weights from a smaller model to a larger one.
    - If dimensions match: Direct copy.
    - If dimensions expand (d_model increase): Center-crop/tile mapping.
    - If new layers exist (growth): Mirror existing ones.
    """
    src_state = src_model.state_dict()
    dst_state = dst_model.state_dict()
    
    print(f"[Migration] Mapping {len(src_state)} src keys to {len(dst_state)} dst keys...")
    
    for name, dst_param in dst_model.named_parameters():
        if name in src_state:
            src_param = src_state[name]
            if src_param.shape == dst_param.shape:
                dst_state[name].copy_(src_param)
                # print(f"  [Direct] {name}")
            else:
                # Dimension Expansion or Compression
                # print(f"  [Scaling] {name}: {src_param.shape} -> {dst_param.shape}")
                
                with torch.no_grad():
                    # Initialize with noise
                    dst_param.data.normal_(mean=0.0, std=0.01)
                    
                    # Compute slices for bidirectional scaling
                    dst_slices = []
                    src_slices = []
                    for s_dim, d_dim in zip(src_param.shape, dst_param.shape):
                        if d_dim >= s_dim:
                            # Expanding: Center source in destination
                            start = (d_dim - s_dim) // 2
                            dst_slices.append(slice(start, start + s_dim))
                            src_slices.append(slice(None))
                        else:
                            # Compressing: Center crop from source to fit destination
                            start = (s_dim - d_dim) // 2
                            dst_slices.append(slice(None))
                            src_slices.append(slice(start, start + d_dim))
                    
                    dst_param.data[tuple(dst_slices)].copy_(src_param[tuple(src_slices)])
        else:
            # Layer Mirroring or New Parameter Initialization
            if "stack.layers" in name:
                try:
                    parts = name.split(".")
                    stack_part = parts[0]
                    layer_idx = int(parts[2])
                    suffix = ".".join(parts[3:])
                    
                    src_layers = []
                    prefix = f"core.{stack_part}.layers."
                    for k in src_state.keys():
                        if k.startswith(prefix):
                            src_layers.append(int(k.split(".")[3]))
                    
                    if src_layers:
                        # For compression, we might want to pick a representative layer
                        # (e.g. median or progressive stride), but mirroring max is fine fallback.
                        max_src_layer = max(src_layers)
                        mirror_key = f"{prefix}{max_src_layer}.{suffix}"
                        if mirror_key in src_state:
                            src_param = src_state[mirror_key]
                            
                            dst_slices = []
                            src_slices = []
                            for s_dim, d_dim in zip(src_param.shape, dst_param.shape):
                                if d_dim >= s_dim:
                                    start = (d_dim - s_dim) // 2
                                    dst_slices.append(slice(start, start + s_dim))
                                    src_slices.append(slice(None))
                                else:
                                    start = (s_dim - d_dim) // 2
                                    dst_slices.append(slice(None))
                                    src_slices.append(slice(start, start + d_dim))
                            dst_param.data[tuple(dst_slices)].copy_(src_param[tuple(src_slices)])
                except Exception as e:
                    print(f"  [Warning] Failed to mirror {name}: {e}")

    dst_model.load_state_dict(dst_state)
    print("[Migration] Success.")

def main():
    parser = argparse.ArgumentParser(description="PCEI Topology Migration (52M -> 1.1B)")
    parser.add_argument("--src", type=str, default="checkpoints/pcei_100m.pt")
    parser.add_argument("--dst", type=str, default="checkpoints/pcei_micro_evolved.pt")
    parser.add_argument("--d-text", type=int, default=256)
    parser.add_argument("--l-text", type=int, default=4)
    parser.add_argument("--d-img", type=int, default=256)
    parser.add_argument("--l-img", type=int, default=2)
    parser.add_argument("--d-audio", type=int, default=128)
    parser.add_argument("--l-audio", type=int, default=2)
    parser.add_argument("--d-text-src", type=int, default=128)
    parser.add_argument("--l-text-src", type=int, default=2)
    parser.add_argument("--d-img-src", type=int, default=128)
    parser.add_argument("--l-img-src", type=int, default=1)
    parser.add_argument("--d-audio-src", type=int, default=64)
    parser.add_argument("--l-audio-src", type=int, default=1)

    args = parser.parse_args()
    
    device = "cpu" # We do migration on CPU to save VRAM
    
    # 1. Load Source
    print(f"[Engine] Loading Source from {args.src}...")
    checkpoint = torch.load(args.src, map_location=device, weights_only=False)
    
    # Initialize Source architecture
    src_nexus = PCEINexus(
        device=device,
        d_model_text=args.d_text_src,
        layers_text=args.l_text_src,
        d_model_img=args.d_img_src,
        layers_img=args.l_img_src,
        d_model_audio=args.d_audio_src,
        layers_audio=args.l_audio_src
    )
    # Patch for heterogeneous stacks
    from src.engine.model import PCEIModel
    class SourceModel(PCEIModel):
        def __init__(self):
            nn.Module.__init__(self)
            from src.core.stack import TextStack, ImageStack, AudioStack
            from src.core.boundary import BoundaryLayer
            from src.engine.seeder import DreamSeeder
            from src.engine.physics import EntanglementBroker
            from src.engine.topology import TopologyShifter
            from src.engine.model import TEXT_SEQ_LEN, IMG_SEQ_LEN, AUDIO_SEQ_LEN
            
            self.text_seq_len = TEXT_SEQ_LEN
            self.img_seq_len = IMG_SEQ_LEN
            self.audio_seq_len = AUDIO_SEQ_LEN
            
            self.text_stack = TextStack(vocab_size=50257, d_model=args.d_text_src, num_layers=args.l_text_src, start_z=0.0)
            self.img_stack = ImageStack(d_model=args.d_img_src, num_layers=args.l_img_src, start_z=2.0)
            self.audio_stack = AudioStack(d_model=args.d_audio_src, num_layers=args.l_audio_src, start_z=4.0)
            
            self.dream_seeder = DreamSeeder(d_model_text=args.d_text_src, d_model_img=args.d_img_src, d_model_audio=args.d_audio_src)
            
            self.boundary_t_i = BoundaryLayer(
                d_model_upper=args.d_text_src, d_model_lower=args.d_img_src,
                seq_len_upper=TEXT_SEQ_LEN, seq_len_lower=IMG_SEQ_LEN
            )
            self.boundary_i_a = BoundaryLayer(
                d_model_upper=args.d_img_src, d_model_lower=args.d_audio_src,
                seq_len_upper=IMG_SEQ_LEN, seq_len_lower=AUDIO_SEQ_LEN
            )
            self.broker = EntanglementBroker()
            self.shifter = TopologyShifter(self)
            self.detectors = {}
            from concurrent.futures import ThreadPoolExecutor
            self.executor = ThreadPoolExecutor(max_workers=3)
            self.dream_injection_scale = 0.1

    src_nexus.core = SourceModel().to(device)
    src_nexus.load_state_dict(checkpoint['model_state_dict'])
    
    # 2. Configure 1.1B Target
    print("[Engine] Initializing 1.1B Target Architecture...")
    # We subclass or monkey-patch the dimensions for the target
    from src.engine.model import PCEIModel
    
    class TargetModel(PCEIModel):
        def __init__(self):
            nn.Module.__init__(self)
            from src.core.stack import TextStack, ImageStack, AudioStack
            from src.core.boundary import BoundaryLayer
            from src.engine.seeder import DreamSeeder
            from src.engine.physics import EntanglementBroker
            from src.engine.topology import TopologyShifter
            from src.engine.model import TEXT_SEQ_LEN, IMG_SEQ_LEN, AUDIO_SEQ_LEN
            
            self.text_seq_len = TEXT_SEQ_LEN
            self.img_seq_len = IMG_SEQ_LEN
            self.audio_seq_len = AUDIO_SEQ_LEN
            
            # Target Dims
            dt, lt = args.d_text, args.l_text
            di, li = args.d_img, args.l_img
            da, la = args.d_audio, args.l_audio
            
            self.text_stack = TextStack(vocab_size=50257, d_model=dt, num_layers=lt, start_z=0.0)
            self.img_stack = ImageStack(d_model=di, num_layers=li, start_z=2.0)
            self.audio_stack = AudioStack(d_model=da, num_layers=la, start_z=4.0)
            
            self.dream_seeder = DreamSeeder(d_model_text=dt, d_model_img=di, d_model_audio=da)
            
            self.boundary_t_i = BoundaryLayer(
                d_model_upper=dt, d_model_lower=di,
                seq_len_upper=TEXT_SEQ_LEN, seq_len_lower=IMG_SEQ_LEN
            )
            self.boundary_i_a = BoundaryLayer(
                d_model_upper=di, d_model_lower=da,
                seq_len_upper=IMG_SEQ_LEN, seq_len_lower=AUDIO_SEQ_LEN
            )
            
            self.broker = EntanglementBroker()
            self.shifter = TopologyShifter(self)
            self.detectors = {}
            from concurrent.futures import ThreadPoolExecutor
            self.executor = ThreadPoolExecutor(max_workers=3)
            self.dream_injection_scale = 0.1
    
    # Initialize Target Nexus with modified model
    dst_nexus = PCEINexus(device=device)
    dst_nexus.core = TargetModel().to(device)
    
    total_params = sum(p.numel() for p in dst_nexus.parameters())
    print(f"[Target] New Goal: {total_params:,} parameters.")
    
    # 3. Migrate
    migrate_weights(src_nexus.core, dst_nexus.core)
    
    # 4. Save
    print(f"[Engine] Saving weights to {args.dst}...")
    torch.save({
        'model_state_dict': dst_nexus.state_dict(),
        'global_step': checkpoint.get('global_step', 0),
    }, args.dst)
    print("[Engine] Evolutionary Leap Initialized.")

if __name__ == "__main__":
    main()
