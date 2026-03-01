# Architecture Diagrams

Three Mermaid diagrams describing the paleoeurope-dem pipeline architecture.
Render with any Mermaid-compatible viewer (GitHub, VSCode Markdown Preview Enhanced, etc.)

---

## 1. Multi-Node Infrastructure

Queue, task manager, network disks (input/code) and network disks (output).

```mermaid
graph TB
    subgraph NET_IN["🗄️  NFS / S3 INPUT  (read-only)"]
        direction LR
        NFS_CODE["📁 /mnt/code\npaleoeurope pkg\nconfigs/"]:::storage
        NFS_FABDEM["📁 /mnt/data/fabdem\nFABDEM v1.2\n~8 000 tiles · 1″ GeoTIFF"]:::storage
        NFS_GEBCO["📁 /mnt/data/gebco\nGEBCO 2024\nglobal NetCDF"]:::storage
        NFS_ICE6G["📁 /mnt/data/ice6g\nICE-6G_C VM5a\nO512 NetCDF"]:::storage
        NFS_EGM["📁 /mnt/data/egm2008\negm2008-25.tif\nundulation grid"]:::storage
    end

    subgraph SUBMIT["🖥️  Submit Node"]
        CLI["run_full_pipeline.py\n--backend celery\n--tile-list tiles.txt\n--epochs 0 6 12 18 21"]:::process
        CHORD["celery.chord\nfusion_group ► paleo_group"]:::process
    end

    subgraph BROKER["⚡  Redis 7  (single instance or Sentinel HA)"]
        Q_FUSION["Queue  db=0\nfusion_tile_task\n↻ FIFO · visibility=600s"]:::queue
        Q_PALEO["Queue  db=0\npaleo_tile_task\n↻ FIFO · visibility=600s"]:::queue
        Q_RESULT["Result store  db=1\nstatus · output_path · error"]:::queue
    end

    subgraph FLOWER["🌸  Flower Monitor  :5555"]
        FL["task list · worker heartbeat\nthroughput graph · retry log"]:::monitor
    end

    subgraph WORKERS["⚙️  Worker Pool  (horizontally scalable)"]
        subgraph W0["Worker-0  (--concurrency 8)"]
            W0T1["Thread 1\nfusion_tile_task\nN51E000"]:::worker
            W0T2["Thread 2\nfusion_tile_task\nN51E001"]:::worker
            W0T3["Thread …\n…"]:::worker
        end
        subgraph W1["Worker-1  (--concurrency 8)"]
            W1T1["Thread 1\npaleo_tile_task\nN51E000 · 21ka"]:::worker
            W1T2["Thread 2\npaleo_tile_task\nN51E001 · 21ka"]:::worker
            W1T3["Thread …\n…"]:::worker
        end
        subgraph WN["Worker-N  …"]
            WNT["Thread …"]:::worker
        end
    end

    subgraph NET_OUT["🗄️  NFS / S3 OUTPUT  (write)"]
        OUT_FUSION["📁 /mnt/output/fusion/\n{TILE}_fusion.tif\nLZW · tiled 512px\nFloat32"]:::storage
        OUT_PALEO["📁 /mnt/output/paleo/{epoch_ka}ka/\n{TILE}_paleo.tif\nLZW · tiled 512px\nFloat32"]:::storage
        OUT_MASK["📁 /mnt/output/masks/\n{TILE}_ice_mask.tif\nUInt8 boolean"]:::storage
    end

    CLI -->|"chord(fusion_group)"| Q_FUSION
    CHORD -->|"callback(paleo_group)"| Q_PALEO
    CLI --> CHORD

    Q_FUSION -->|"pop task"| W0
    Q_FUSION -->|"pop task"| W1
    Q_PALEO  -->|"pop task"| W0
    Q_PALEO  -->|"pop task"| W1
    Q_PALEO  -->|"pop task"| WN

    W0 -->|"ack + result"| Q_RESULT
    W1 -->|"ack + result"| Q_RESULT

    Q_RESULT -.->|"poll"| CLI
    BROKER -.->|"events"| FLOWER

    NFS_CODE   -->|"pip install -e ."| W0
    NFS_CODE   -->|"pip install -e ."| W1
    NFS_FABDEM -->|"read window"| W0
    NFS_GEBCO  -->|"read window"| W0
    NFS_EGM    -->|"read undulation"| W0
    NFS_ICE6G  -->|"read delta"| W1
    NFS_FABDEM -->|"read window"| W1

    W0 -->|"write GeoTIFF"| OUT_FUSION
    W1 -->|"write GeoTIFF"| OUT_PALEO
    W1 -->|"write mask"| OUT_MASK

    classDef storage fill:#2d4a6e,stroke:#5b8dd9,color:#e8f0fe
    classDef process fill:#1e4a2e,stroke:#4caf50,color:#e8f5e9
    classDef queue   fill:#4a2d00,stroke:#ff9800,color:#fff3e0
    classDef worker  fill:#2d1a4a,stroke:#9c27b0,color:#f3e5f5
    classDef monitor fill:#4a1a1a,stroke:#f44336,color:#ffebee
```

---

## 2. ETL Pipeline — Atomic Algorithm

Block diagram of the full pipeline at the level of individual function calls and array operations.

```mermaid
flowchart TD
    START([" tile_id  ·  epoch_ka "])

    subgraph STAGE1["STAGE 1 — EXTRACT  (per tile)"]
        S1A["FabdemLoader.read_window(tile_id)\n──────────────────────────────\n① parse SRTM naming → bbox\n② open GeoTIFF with rasterio window\n③ resample to target resolution if needed\n④ mask nodata → np.nan\nout: z_fab  [H×W float32]  EGM2008 ortho height"]:::extract

        S1B["GebcoLoader.read_window(bbox)\n──────────────────────────────\n① open GEBCO NetCDF via xarray\n② slice lat/lon window\n③ TID mask: keep TID∈{0,1,10,11} only\n   (direct soundings + predicted)\n④ reproject WGS84 → tile CRS\nout: z_geb  [H×W float32]  MSL depth/height"]:::extract

        S1C["Optional: DatumCorrector.align(z_geb, bbox)\n──────────────────────────────\n(enabled only when validated)\n① open egm2008-25.tif\n② RegularGridInterpolator(φ,λ) → N(φ,λ)\n③ if enabled:  z_geb_adj = z_geb − N\n   else:       z_geb_adj = z_geb\nout: z_geb_adj  [H×W float32]\n     N_grid      [H×W float32]  undulation"]:::extract
    end

    subgraph STAGE2["STAGE 2 — TRANSFORM A  (fusion)"]
        S2A["Coastline mask\n──────────────────────────────\n① z_fab > 0  →  land_mask  [bool]\n② z_geb_adj ≤ 0  →  ocean_mask  [bool]\n③ conflict pixels → FABDEM wins\nout: land_mask, ocean_mask"]:::transform

        S2B["RasterBlender.compute_alpha(land_mask)\n──────────────────────────────\n① EDT(land_mask) → d_land  [px]\n② EDT(~land_mask) → d_ocean [px]\n③ latitude correction:\n   aspect = (1.0, cos(φ_centre))\n④ signed_dist = d_land − d_ocean  [px]\n⑤ α = clip((signed_dist + D) / (2D), 0, 1)\n   D = blend_distance_px (default 50)\nout: α  [H×W float32]  ∈ [0, 1]"]:::transform

        S2C["RasterBlender.blend(z_fab, z_geb_adj, α)\n──────────────────────────────\n① short-circuit: α≡1 → return z_fab\n② short-circuit: α≡0 → return z_geb_adj\n③ z_fused = α·z_fab + (1−α)·z_geb_adj\nout: z_fused  [H×W float32]"]:::transform
    end

    subgraph STAGE3["STAGE 3 — TRANSFORM B  (GIA)"]
        S3A["Ice6gLoader.get_fields(epoch_ka)\n──────────────────────────────\n① open ICE-6G_C VM5a NetCDF\n② detect dims: lat/lon/time (flexible)\n③ time axis: positive or negative ka\n④ interpolate to nearest epoch\nout: sice  [Φ×Λ float32]  ice thickness [m]\n     dz    [Φ×Λ float32]  bedrock deform [m]\n     rsl   [Φ×Λ float32]  relative sea level [m]"]:::gia

        S3B["apply_gia_delta(z_fused, bbox, epoch_ka)\n──────────────────────────────\n① build RegularGridInterpolator on\n   dz(φ,λ) from ICE-6G grid\n② make pixel coords φ_ij, λ_ij\n③ Δz = interpolator(φ_ij, λ_ij)\n④ z_paleo = z_fused + Δz\nout: z_paleo  [H×W float32]\n     ice_mask [H×W bool]  sice>0 pixels"]:::gia

        S3C["IceEnvelope.surface_elevation(z_paleo, bbox)\n──────────────────────────────\n① where ice_mask==True:\n   z_surface = z_paleo + ice_height\n② bedrock_delta applied under ice\nout: z_full  [H×W float32]\n     (NaN where open ocean, ice→ice surface)"]:::gia
    end

    subgraph STAGE4["STAGE 4 — LOAD  (write outputs)"]
        S4A["write_geotiff(z_fused, path_fusion)\n──────────────────────────────\n driver=GTiff · dtype=float32\n compress=LZW · tiled=YES\n blockxsize=512 · blockysize=512\n EPSG:4326  1″ pixel size"]:::load

        S4B["write_geotiff(z_full, path_paleo)\n+ write_geotiff(ice_mask.uint8, path_mask)\n──────────────────────────────\n subdir: paleo/{epoch_ka}ka/\n ice_mask: UInt8 · 0/1"]:::load
    end

    END(["✓  tile done\nresult → Redis db=1"])

    START --> S1A & S1B
    S1B --> S1C
    S1A --> S2A
    S1C --> S2A
    S2A --> S2B
    S2B --> S2C
    S1A --> S2C
    S1C --> S2C
    S2C --> S4A
    S2C --> S3A
    S3A --> S3B
    S3B --> S3C
    S3C --> S4B
    S4A & S4B --> END

    classDef extract   fill:#1a3a5c,stroke:#5b8dd9,color:#e8f0fe
    classDef transform fill:#1a3a1a,stroke:#4caf50,color:#e8f5e9
    classDef gia       fill:#3a1a3a,stroke:#ce93d8,color:#fce4ec
    classDef load      fill:#3a2a00,stroke:#ffb300,color:#fffde7
```

---

## 3. Data Types, Correction Matrices & Layer Superposition

Verified sources with citations, where each correction matrix is built, and how layers are composited pixel-by-pixel.

```mermaid
graph TB
    subgraph SRC["PRIMARY DATA SOURCES  (verified, citable)"]
        direction LR
        D1["🟦 FABDEM v1.2\nFloat32 GeoTIFF  1 arc-sec (~30 m)\n──────────────────────────────────\n• Canopy + building removed from SRTM\n• Vertical datum: EGM2008 (already)\n• Coverage: 80°S – 80°N land\n• doi:10.5194/essd-14-4677-2022\n  Hawker et al. 2022"]:::src_fab

        D2["🟩 GEBCO 2024\nInt16 NetCDF  15 arc-sec (~450 m)\n──────────────────────────────────\n• Global ocean + land DEM\n• Vertical datum: MSL\n• TID: type-identifier per pixel\n• doi:10.5285/1c44ce99-…\n  GEBCO Compilation Group 2024"]:::src_geb

        D3["🟨 EGM2008\nFloat32 GeoTIFF  2.5 arc-min (~4.6 km)\n──────────────────────────────────\n• Geoid undulation N(φ,λ)  [m]\n• Global: -107 m … +86 m\n• Public domain (US Gov)\n• doi:10.1029/2011JB008916\n  Pavlis et al. 2012"]:::src_egm

        D4["🟪 ICE-6G_C VM5a\nFloat32 NetCDF  0.5° (~55 km)\n──────────────────────────────────\n• sice: ice thickness  [m]\n• stgr: bedrock deform (GIA)  [m]\n• rsl:  relative sea level  [m]\n• Time: 0 – 26 ka BP · 1 ka steps\n• doi:10.1002/2014JB011176\n  Peltier et al. 2015"]:::src_ice
    end

    subgraph MATRIX["CORRECTION MATRIX  N(φ,λ)  — how it is built"]
        direction TB
        M1["Input: egm2008-25.tif\n────────────────────\nrasterio.open() → band 1\nshape: [Φ_g × Λ_g]\nvalues: undulation N in metres\nCRS: EPSG:4326"]:::matrix_in

        M2["scipy.interpolate\n.RegularGridInterpolator\n────────────────────\npoints = (lat_1d, lon_1d)\nvalues = N_grid  [Φ_g × Λ_g]\nmethod = 'linear'"]:::matrix_build

        M3["per pixel (φ_ij, λ_ij):\nN_ij = interp((φ_ij, λ_ij))\n────────────────────\nshape matches tile: [H × W]\nbuilt once per tile,\nreused for all GEBCO pixels"]:::matrix_out

        M4["Optional offset (if enabled):\nz_geb_adj(i,j) = z_geb(i,j) − N(i,j)\nelse: z_geb_adj = z_geb\n────────────────────\nValidation-dependent vertical\nalignment step"]:::matrix_apply

        M1 --> M2 --> M3 --> M4
    end

    subgraph GIA_MATRIX["GIA DELTA MATRIX  Δz(φ,λ,t)  — how it is built"]
        direction TB
        G1["Input: ICE-6G_C VM5a\n────────────────────\nxarray.open_dataset()\nvariables: stgr, sice, rsl\ndims: lat[360] × lon[720] × time[27]"]:::gia_in

        G2["epoch selection:\nt_idx = argmin|time − epoch_ka|\nslice: ds.stgr.isel(time=t_idx)\n────────────────────\nshape: [360 × 720]  0.5° grid"]:::gia_build

        G3["scipy.interpolate\n.RegularGridInterpolator\n────────────────────\npoints = (lat_0.5, lon_0.5)\nvalues = stgr  [360 × 720]\nmethod = 'linear'"]:::gia_build

        G4["per pixel (φ_ij, λ_ij):\nΔz_ij = interp((φ_ij, λ_ij))\n────────────────────\nshape [H × W]  tile resolution\nRange: −200 m … +500 m\n(Scandinavia uplift max ~300 m)"]:::gia_out

        G1 --> G2 --> G3 --> G4
    end

    subgraph SUPER["LAYER SUPERPOSITION  — pixel-level algebra"]
        direction TB
        L1["Layer 0 · z_fab  [H×W]\nFABDEM bare-earth\nland only (ocean = NaN)\nDatum: EGM2008"]:::layer_fab

        L2["Layer 1 · z_geb_adj  [H×W]\nGEBCO (optional offset)\n= z_geb (or z_geb − N_ij)\nDatum: (validation-dependent)"]:::layer_geb

        L3["Layer 2 · α  [H×W]  ∈ [0,1]\nBlend weight from EDT\nα=1 → pure FABDEM (land interior)\nα=0 → pure GEBCO (ocean interior)\nCoastline transition: 50 px ramp"]:::layer_alpha

        L4["Layer 3 · z_fused  [H×W]\n= α·z_fab + (1−α)·z_geb_adj\n────────────────────────────\nSingle continuous surface:\nland + coast + ocean\nDatum: (working reference)  Resolution: 1″"]:::layer_fused

        L5["Layer 4 · Δz_GIA  [H×W]\nBedrock deformation at epoch t\nfrom ICE-6G interpolated to 1″\nRange: −200 … +500 m"]:::layer_gia

        L6["Layer 5 · ice_mask  [H×W bool]\nsice_ij > 0  →  True\nTrue pixels: elevation set to\nz_paleo + ice_height\nFalse pixels: unchanged"]:::layer_ice

        LF["FINAL · z_paleo  [H×W]\n= z_fused(i,j) + Δz_GIA(i,j)\n─────────────────────────────────\nwhere ice_mask==True:\n  z_paleo = z_paleo + ice_height\nwhere open_ocean==True (RSL change):\n  z_paleo = NaN\n─────────────────────────────────\nOutput: Float32 GeoTIFF  1 arc-sec\nDatum: EGM2008  Epoch: t ka BP"]:::layer_final

        L1 & L2 --> L3
        L3 --> L4
        L4 --> L5
        L5 & L6 --> LF
    end

    D1 -->|"read_window(tile_id)\nbbox → rasterio"| L1
    D2 -->|"TID-masked window\nxarray slice"| L2
    D3 -->|"RegularGridInterp\nN(φ,λ) matrix"| MATRIX
    MATRIX --> L2
    D4 -->|"epoch slice\nRegularGridInterp"| GIA_MATRIX
    GIA_MATRIX --> L5
    GIA_MATRIX --> L6

    classDef src_fab  fill:#1a3050,stroke:#5b8dd9,color:#bbdefb
    classDef src_geb  fill:#1a3a1a,stroke:#66bb6a,color:#c8e6c9
    classDef src_egm  fill:#3a2e00,stroke:#ffc107,color:#fff9c4
    classDef src_ice  fill:#2a1a3a,stroke:#ba68c8,color:#e1bee7
    classDef matrix_in    fill:#0d2137,stroke:#4fc3f7,color:#e1f5fe
    classDef matrix_build fill:#0d2137,stroke:#29b6f6,color:#e1f5fe
    classDef matrix_out   fill:#0d2137,stroke:#0288d1,color:#e1f5fe
    classDef matrix_apply fill:#0d2137,stroke:#0277bd,color:#b3e5fc
    classDef gia_in    fill:#1a0d2e,stroke:#ab47bc,color:#f3e5f5
    classDef gia_build fill:#1a0d2e,stroke:#9c27b0,color:#f3e5f5
    classDef gia_out   fill:#1a0d2e,stroke:#7b1fa2,color:#e1bee7
    classDef layer_fab    fill:#163050,stroke:#5b8dd9,color:#bbdefb
    classDef layer_geb    fill:#163a16,stroke:#66bb6a,color:#c8e6c9
    classDef layer_alpha  fill:#2a1e00,stroke:#ffb300,color:#fff8e1
    classDef layer_fused  fill:#1a2a1a,stroke:#43a047,color:#dcedc8
    classDef layer_gia    fill:#200d2e,stroke:#ba68c8,color:#e1bee7
    classDef layer_ice    fill:#001a2e,stroke:#4dd0e1,color:#e0f7fa
    classDef layer_final  fill:#2e1000,stroke:#ff7043,color:#fbe9e7
```
