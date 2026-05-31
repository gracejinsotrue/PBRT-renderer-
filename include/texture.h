#pragma once

#include <color.h>
#include <common.h>
#include <filesystem/resolver.h>
#include <stb_image.h>
#include <vector>
#include <cmath>

NORI_NAMESPACE_BEGIN

class Texture2D
{
public:
    Texture2D(const std::string &filename)
    {
        filesystem::path resolved = getFileResolver()->resolve(filename);
        int channels;
        float *pixels = stbi_loadf(resolved.str().c_str(),
                                   &m_width, &m_height, &channels, 3);
        if (!pixels)
            throw NoriException("Texture2D: failed to load \"%s\"", filename);

        m_data.resize(m_width * m_height);
        for (int i = 0; i < m_width * m_height; ++i)
            m_data[i] = Color3f(pixels[3 * i], pixels[3 * i + 1], pixels[3 * i + 2]);

        stbi_image_free(pixels);
        std::cout << "Loaded texture \"" << filename
                  << "\" (" << m_width << "x" << m_height << ")" << std::endl;
    }

    Color3f eval(const Point2f &uv) const
    {

        float u = uv.x() - std::floor(uv.x());
        float v = uv.y() - std::floor(uv.y());

        v = 1.0f - v;

        float x = u * m_width - 0.5f;
        float y = v * m_height - 0.5f;

        int x0 = (int)std::floor(x);
        int y0 = (int)std::floor(y);
        float fx = x - x0;
        float fy = y - y0;

        auto fetch = [&](int ix, int iy) -> Color3f
        {
            ix = ((ix % m_width) + m_width) % m_width;
            iy = ((iy % m_height) + m_height) % m_height;
            return m_data[iy * m_width + ix];
        };

        return (1 - fx) * (1 - fy) * fetch(x0, y0) + fx * (1 - fy) * fetch(x0 + 1, y0) + (1 - fx) * fy * fetch(x0, y0 + 1) + fx * fy * fetch(x0 + 1, y0 + 1);
    }

private:
    std::vector<Color3f> m_data;
    int m_width, m_height;
};

/// Single-channel alpha texture for alpha masking , for things like eyelash and eyebrow
/// Loads the alpha channel, which is the 4th component from an RGBA image.
class AlphaTexture
{
public:
    AlphaTexture(const std::string &filename)
    {
        filesystem::path resolved = getFileResolver()->resolve(filename);
        int channels;
        // request 4 channels so we get the alpha even from RGB sources
        float *pixels = stbi_loadf(resolved.str().c_str(),
                                   &m_width, &m_height, &channels, 4);
        if (!pixels)
            throw NoriException("AlphaTexture: failed to load \"%s\"", filename);

        m_data.resize(m_width * m_height);
        for (int i = 0; i < m_width * m_height; ++i)
            m_data[i] = pixels[4 * i + 3]; // store only the alpha channel

        stbi_image_free(pixels);
        std::cout << "Loaded alpha texture \"" << filename
                  << "\" (" << m_width << "x" << m_height << ")" << std::endl;
    }

    float eval(const Point2f &uv) const
    {
        float u = uv.x() - std::floor(uv.x());
        float v = uv.y() - std::floor(uv.y());

        v = 1.0f - v;

        float x = u * m_width - 0.5f;
        float y = v * m_height - 0.5f;

        int x0 = (int)std::floor(x);
        int y0 = (int)std::floor(y);
        float fx = x - x0;
        float fy = y - y0;

        auto fetch = [&](int ix, int iy) -> float
        {
            ix = ((ix % m_width) + m_width) % m_width;
            iy = ((iy % m_height) + m_height) % m_height;
            return m_data[iy * m_width + ix];
        };

        return (1 - fx) * (1 - fy) * fetch(x0, y0) + fx * (1 - fy) * fetch(x0 + 1, y0) + (1 - fx) * fy * fetch(x0, y0 + 1) + fx * fy * fetch(x0 + 1, y0 + 1);
    }

private:
    std::vector<float> m_data;
    int m_width, m_height;
};

NORI_NAMESPACE_END