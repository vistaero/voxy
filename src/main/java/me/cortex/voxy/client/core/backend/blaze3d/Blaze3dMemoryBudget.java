package me.cortex.voxy.client.core.backend.blaze3d;

import java.util.Locale;
import me.cortex.voxy.common.Logger;
import oshi.SystemInfo;

/**
 * Estimates the expanded-memory cost of the Blaze3D renderer. Cortex's native renderer keeps
 * each quad packed into eight bytes; Blaze3D expands it into four 28-byte vertices, so its memory
 * requirements need to follow render distance and screen-space quality instead of a fixed cap.
 */
public final class Blaze3dMemoryBudget {
    private static final long MIB = 1024L * 1024L;
    private static final long GIB = 1024L * MIB;
    private static final long MIN_GEOMETRY_BUDGET = 512L * MIB;
    private static final long MAX_GEOMETRY_BUDGET = 8L * GIB;
    private static final long MIN_STAGING_BUDGET = 128L * MIB;
    private static final long MAX_STAGING_BUDGET = 768L * MIB;
    // Full 65,536-model RGBA atlas including the four allocated mip levels, plus render targets,
    // the shared index buffer and small fixed GPU resources.
    private static final long FIXED_VRAM_BYTES = 576L * MIB;
    // Captured Minecraft atlas, baked-model upload results and CPU metadata during population.
    private static final long FIXED_RAM_BYTES = 256L * MIB;

    private Blaze3dMemoryBudget() {
    }

    /** Total VRAM is a capacity hint, not a measurement of currently free device memory. */
    public static long detectGeometryLimit(String deviceName) {
        long limit = 512L * MIB;
        String active = normalizeDeviceName(deviceName);
        try {
            long capacity = Long.MAX_VALUE;
            for (var card : new SystemInfo().getHardware().getGraphicsCards()) {
                String name = normalizeDeviceName(card.getName());
                if (!name.isEmpty() && !active.isEmpty()
                        && (active.contains(name) || name.contains(active)) && card.getVRam() > 0L) {
                    capacity = Math.min(capacity, card.getVRam());
                }
            }
            if (capacity != Long.MAX_VALUE) {
                // Leave half to Minecraft, Sodium, shader packs and the desktop. Our atlas and
                // render targets consume part of the remaining half before any geometry fits.
                limit = clamp(capacity / 2L - FIXED_VRAM_BYTES, 64L * MIB, MAX_GEOMETRY_BUDGET);
            }
        } catch (RuntimeException | LinkageError exception) {
            Logger.warn("Unable to determine Blaze3D device capacity; using a 512 MiB geometry limit: "
                    + exception.getMessage());
        }
        Logger.info("Blaze3D geometry safety limit for " + deviceName + ": " + formatBytes(limit)
                + " (capacity estimate; allocation failures reduce it further).");
        return limit;
    }

    private static String normalizeDeviceName(String name) {
        return name.toLowerCase(Locale.ROOT).replaceAll("[^a-z0-9]", "");
    }

    public static Estimate estimate(float sectionRenderDistance, float subdivisionSize) {
        double renderDistanceChunks = Math.max(20.0, sectionRenderDistance * 32.0);
        double safeSubdivisionSize = Math.max(28.0, subdivisionSize);

        // Calibrated against a 2,592-block (162 chunk), 64-pixel capture. Distance grows slower
        // than area because progressively farther rings select progressively coarser Cortex LoDs.
        double distanceFactor = Math.pow(renderDistanceChunks / 162.0, 0.80);
        double qualityFactor = Math.pow(64.0 / safeSubdivisionSize, 1.35);
        long steadyGeometryBytes = clamp(Math.round(2.60 * GIB * distanceFactor * qualityFactor),
                256L * MIB, 32L * GIB);

        // Branch hand-offs temporarily retain a coarser parent while its descendants finish.
        long requestedGeometryBudget = Math.round(steadyGeometryBytes * 1.20);
        long geometryBudgetBytes = clamp(requestedGeometryBudget, MIN_GEOMETRY_BUDGET, MAX_GEOMETRY_BUDGET);
        long stagingBudgetBytes = clamp(Math.round(steadyGeometryBytes * 0.12),
                MIN_STAGING_BUDGET, MAX_STAGING_BUDGET);
        long requiredVramBytes = requestedGeometryBudget + FIXED_VRAM_BYTES;
        long requiredRamBytes = stagingBudgetBytes + FIXED_RAM_BYTES;
        return new Estimate(requiredRamBytes, requiredVramBytes, geometryBudgetBytes, stagingBudgetBytes,
                requestedGeometryBudget > MAX_GEOMETRY_BUDGET);
    }

    public static String formatBytes(long bytes) {
        if (bytes >= GIB) {
            return String.format(Locale.ROOT, "%.1f GiB", bytes / (double) GIB);
        }
        return Math.round(bytes / (double) MIB) + " MiB";
    }

    private static long clamp(long value, long minimum, long maximum) {
        return Math.max(minimum, Math.min(maximum, value));
    }

    public record Estimate(long requiredRamBytes,
                           long requiredVramBytes,
                           long geometryBudgetBytes,
                           long stagingBudgetBytes,
                           boolean exceedsSafetyLimit) {
        public String shortDescription() {
            return "~" + formatBytes(this.requiredVramBytes) + " VRAM / ~"
                    + formatBytes(this.requiredRamBytes) + " RAM";
        }
    }
}
