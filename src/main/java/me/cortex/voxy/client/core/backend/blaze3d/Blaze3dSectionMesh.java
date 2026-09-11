package me.cortex.voxy.client.core.backend.blaze3d;

import me.cortex.voxy.client.core.rendering.building.BuiltSection;
import me.cortex.voxy.common.world.WorldEngine;
import net.minecraft.world.level.CardinalLighting;
import org.jetbrains.annotations.Nullable;
import org.lwjgl.system.MemoryUtil;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/** Expands Cortex's packed BuiltSection quads into Blaze3D's backend-neutral vertex format. */
final class Blaze3dSectionMesh implements AutoCloseable {
    private static final int VERTEX_STRIDE = 28; // DefaultVertexFormat.POSITION_TEX_LIGHTMAP_COLOR
    private static final int VERTICES_PER_QUAD = 4;
    private static final float FACE_EPSILON = 0.00005f;

    private final long position;
    private final @Nullable ByteBuffer opaqueVertices;
    private final int opaqueQuadCount;
    private final @Nullable ByteBuffer translucentVertices;
    private final int translucentQuadCount;
    private final long requiredTextureVersion;

    private Blaze3dSectionMesh(long position,
                              @Nullable ByteBuffer opaqueVertices, int opaqueQuadCount,
                              @Nullable ByteBuffer translucentVertices, int translucentQuadCount,
                              long requiredTextureVersion) {
        this.position = position;
        this.opaqueVertices = opaqueVertices;
        this.opaqueQuadCount = opaqueQuadCount;
        this.translucentVertices = translucentVertices;
        this.translucentQuadCount = translucentQuadCount;
        this.requiredTextureVersion = requiredTextureVersion;
    }

    static Blaze3dSectionMesh expand(BuiltSection section, Blaze3dModelStore models, CardinalLighting lighting) {
        if (section.isEmpty()) {
            return new Blaze3dSectionMesh(section.position, null, 0, null, 0, 0L);
        }

        int totalQuads = (int) (section.geometryBuffer.size / Long.BYTES);
        int translucentQuads = section.offsets[1] - section.offsets[0];
        int opaqueQuads = totalQuads - translucentQuads;
        ByteBuffer translucent = null;
        ByteBuffer opaque = null;
        try {
            translucent = allocateVertices(translucentQuads);
            opaque = allocateVertices(opaqueQuads);
            long source = section.geometryBuffer.address;
            long requiredTextureVersion = 0L;
            for (int index = 0; index < translucentQuads; index++) {
                requiredTextureVersion = Math.max(requiredTextureVersion,
                        emitQuad(translucent, MemoryUtil.memGetLong(source + (long) index * Long.BYTES),
                                section.position, models, lighting));
            }
            for (int index = translucentQuads; index < totalQuads; index++) {
                requiredTextureVersion = Math.max(requiredTextureVersion,
                        emitQuad(opaque, MemoryUtil.memGetLong(source + (long) index * Long.BYTES),
                                section.position, models, lighting));
            }
            if (translucent != null) translucent.flip();
            if (opaque != null) opaque.flip();
            return new Blaze3dSectionMesh(section.position, opaque, opaqueQuads, translucent, translucentQuads,
                    requiredTextureVersion);
        } catch (RuntimeException | OutOfMemoryError exception) {
            free(opaque);
            free(translucent);
            throw exception;
        }
    }

    private static @Nullable ByteBuffer allocateVertices(int quadCount) {
        if (quadCount == 0) {
            return null;
        }
        long byteCount = (long) quadCount * VERTICES_PER_QUAD * VERTEX_STRIDE;
        if (byteCount > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Expanded LoD section is too large: " + byteCount + " bytes");
        }
        return MemoryUtil.memAlloc((int) byteCount).order(ByteOrder.nativeOrder());
    }

    private static long emitQuad(ByteBuffer destination, long quad, long sectionPosition,
                                 Blaze3dModelStore models, CardinalLighting lighting) {
        int face = (int) (quad & 7L);
        int axis = face >> 1;
        int modelId = (int) ((quad >>> 26) & 0xFFFFL);
        int biomeId = (int) ((quad >>> 46) & 0x1FFL);
        int light = (int) ((quad >>> 55) & 0xFFL);
        int modelFlags = models.modelFlags(modelId);
        int faceData = models.faceData(modelId, face);
        boolean fluid = (modelFlags & 16) != 0;

        float textureMinU = (faceData & 15) / 16.0f - FACE_EPSILON;
        float textureEndU = ((faceData >>> 4) & 15) / 16.0f + 1.0f / 16.0f;
        float textureMinV = ((faceData >>> 8) & 15) / 16.0f - FACE_EPSILON;
        float textureEndV = ((faceData >>> 12) & 15) / 16.0f + 1.0f / 16.0f;
        float textureSizeU = textureEndU - textureMinU;
        float textureSizeV = textureEndV - textureMinV;

        float geometryMinU = fluid ? 0.0f : textureMinU;
        float geometryMinV = fluid ? 0.0f : textureMinV;
        float geometrySizeU = fluid ? 1.0f : textureSizeU;
        float geometrySizeV = fluid ? 1.0f : textureSizeV;

        int quadSizeU = fluid ? 1 : (int) ((quad >>> 3) & 15L) + 1;
        int quadSizeV = fluid ? 1 : (int) ((quad >>> 7) & 15L) + 1;
        geometrySizeU += quadSizeU - 1;
        geometrySizeV += quadSizeV - 1;
        textureSizeU += quadSizeU - 1;
        textureSizeV += quadSizeV - 1;

        int encodedDepth = (faceData >>> 16) & 63;
        if (encodedDepth == 63) encodedDepth = 64;
        float depth = fluid ? 0.0f : encodedDepth / 64.0f;
        if ((face & 1) != 0) depth = 1.0f - depth;

        float localX = (quad >>> 21) & 31L;
        float localY = (quad >>> 16) & 31L;
        float localZ = (quad >>> 11) & 31L;
        int lodLevel = WorldEngine.getLevel(sectionPosition);
        float scale = 1 << lodLevel;
        float baseX = WorldEngine.getX(sectionPosition) * 32.0f * scale + localX * scale;
        float baseY = WorldEngine.getY(sectionPosition) * 32.0f * scale + localY * scale;
        float baseZ = WorldEngine.getZ(sectionPosition) * 32.0f * scale + localZ * scale;
        if (axis == 0) {
            baseX += geometryMinU * scale;
            baseY += depth * scale;
            baseZ += geometryMinV * scale;
        } else if (axis == 1) {
            baseX += geometryMinU * scale;
            baseY += geometryMinV * scale;
            baseZ += depth * scale;
        } else {
            baseX += depth * scale;
            baseY += geometryMinU * scale;
            baseZ += geometryMinV * scale;
        }

        int tint = models.tintColour(modelId, biomeId);
        if (tint == -1) tint = 0xFFFF_FFFF;
        int tintState = (faceData >>> 24) & 3;
        boolean shaded = (modelFlags & 8) != 0;
        float directional = directionalTint(lighting, shaded, face);
        int shade = Math.clamp(Math.round(directional * 255.0f), 0, 255);
        int packedFlags = light | (face << 8) | (tintState << 11);
        if (((faceData >>> 22) & 1) != 0
                || (((faceData >>> 23) & 1) != 0 && (quadSizeU > 1 || quadSizeV > 1))) {
            packedFlags |= 1 << 13;
        }
        if ((modelFlags & 4) != 0) {
            packedFlags |= 1 << 14;
        }

        int fluidHeights = (int) ((quad >>> 3) & 0xFFL) | (int) (((quad >>> 42) & 15L) << 8);
        for (int corner = 0; corner < 4; corner++) {
            int cornerU = (corner >>> 1) & 1;
            int cornerV = corner & 1;
            float cornerOffsetU = geometrySizeU * cornerU * scale;
            float cornerOffsetV = geometrySizeV * cornerV * scale;
            float x = baseX;
            float y = baseY;
            float z = baseZ;
            if (axis == 0) {
                x += cornerOffsetU;
                z += cornerOffsetV;
            } else if (axis == 1) {
                x += cornerOffsetU;
                y += cornerOffsetV;
            } else {
                y += cornerOffsetU;
                z += cornerOffsetV;
            }
            if (fluid) {
                y += fluidHeightOffset(face, axis, cornerU, cornerV, fluidHeights) * scale;
            }
            float u = textureMinU + textureSizeU * cornerU;
            float v = textureMinV + textureSizeV * cornerV;
            putVertex(destination, x, y, z, tint, u, v, modelId, packedFlags, shade);
        }
        return models.textureVersion(modelId);
    }

    private static float fluidHeightOffset(int face, int axis, int cornerU, int cornerV, int heights) {
        int heightIndex = -1;
        if (face == 1) {
            heightIndex = (cornerU << 1) | cornerV;
        } else if (axis == 1 && cornerV == 1) {
            heightIndex = (cornerU << 1) | (face & 1);
        } else if (axis == 2 && cornerU == 1) {
            heightIndex = ((face & 1) << 1) | cornerV;
        }
        if (heightIndex == -1) {
            return 0.0f;
        }
        return (((heights >>> (heightIndex * 3)) & 7) + 1) / 8.0f - 1.0f;
    }

    private static float directionalTint(CardinalLighting lighting, boolean shaded, int face) {
        if (!shaded) return lighting.up();
        return switch (face >> 1) {
            case 1 -> lighting.north();
            case 2 -> lighting.east();
            default -> face == 1 ? lighting.up() : lighting.down();
        };
    }

    private static void putVertex(ByteBuffer out, float x, float y, float z, int tint,
                                  float u, float v, int modelId, int flags, int shade) {
        out.putFloat(x).putFloat(y).putFloat(z);
        out.putFloat(u).putFloat(v);
        out.putShort((short) modelId).putShort((short) flags);
        out.put((byte) (tint >>> 16)).put((byte) (tint >>> 8)).put((byte) tint).put((byte) shade);
    }

    long position() {
        return this.position;
    }

    @Nullable ByteBuffer opaqueVertices() {
        return this.opaqueVertices;
    }

    int opaqueQuadCount() {
        return this.opaqueQuadCount;
    }

    @Nullable ByteBuffer translucentVertices() {
        return this.translucentVertices;
    }

    int translucentQuadCount() {
        return this.translucentQuadCount;
    }

    long requiredTextureVersion() {
        return this.requiredTextureVersion;
    }

    long vertexBytes() {
        return (long) (this.opaqueQuadCount + this.translucentQuadCount)
                * VERTICES_PER_QUAD * VERTEX_STRIDE;
    }

    boolean isEmpty() {
        return this.opaqueQuadCount == 0 && this.translucentQuadCount == 0;
    }

    @Override
    public void close() {
        free(this.opaqueVertices);
        free(this.translucentVertices);
    }

    private static void free(@Nullable ByteBuffer buffer) {
        if (buffer != null) {
            MemoryUtil.memFree(buffer);
        }
    }
}
