import { useState, useEffect } from 'react'
import { productImageUrl } from '../api/client'

export default function ProductPanel({ sceneId, products, compareMode, tasks, onSelectProduct }) {
  const [activeTab, setActiveTab] = useState('products')
  const [selected, setSelected] = useState([])

  useEffect(() => {
    setSelected([])
  }, [sceneId])

  const previewProducts = (products || []).filter((p) =>
    p.output_paths?.some((path) =>
      path.endsWith('.png') || path.endsWith('.tif')
    )
  )

  function toggleSelect(product) {
    const isAlreadySelected = selected.find((p) => p.product_id === product.product_id)
    
    if (isAlreadySelected) {
      onSelectProduct?.(null)
      setSelected((prev) => prev.filter((p) => p.product_id !== product.product_id))
    } else {
      if (compareMode && selected.length >= 2) return
      onSelectProduct?.(product)
      setSelected((prev) => [...prev, product])
    }
  }

  if (!sceneId) return null

  return (
    <div className="product-drawer">
      <div className="tabs">
        <button
          className={`tab ${activeTab === 'products' ? 'active' : ''}`}
          onClick={() => setActiveTab('products')}
        >
          products
        </button>
        {compareMode && (
          <button
            className={`tab ${activeTab === 'compare' ? 'active' : ''}`}
            onClick={() => setActiveTab('compare')}
          >
            compare {selected.length > 0 ? `(${selected.length})` : ''}
          </button>
        )}
        <button
          className={`tab ${activeTab === 'temporal' ? 'active' : ''}`}
          onClick={() => setActiveTab('temporal')}
        >
          temporal
        </button>
        <span style={{ marginLeft: 'auto', padding: '8px 12px', fontSize: 11, color: '#666' }}>
          {sceneId}
        </span>
      </div>

      <div className="tab-content">
        {activeTab === 'products' && (
          <div className="product-grid">
            {previewProducts.length === 0 && (
              <div className="empty-state">no previews available</div>
            )}
            {previewProducts.map((product) => {
              const previewPath = product.output_paths?.find((p) => p.endsWith('.png') || p.endsWith('.tif'))
              const isSelected = selected.find(
                (p) => p.product_id === product.product_id
              )
              return (
                <div
                  key={product.product_id}
                  className={`product-card ${isSelected ? 'selected' : ''}`}
                  onClick={() => toggleSelect(product)}
                >
                  <img
                    src={productImageUrl(previewPath)}
                    alt={product.task}
                    loading="lazy"
                  />
                  <div className="product-label">
                    {product.task.replace(/_/g, ' ')}
                    {' · '}
                    {product.completed_at?.slice(0, 10) || 'unknown date'}
                  </div>
                </div>
              )
            })}
          </div>
        )}

        {activeTab === 'compare' && (
          <CompareView selected={selected} />
        )}

        {activeTab === 'temporal' && (
          <TemporalView sceneId={sceneId} products={previewProducts} />
        )}
      </div>
    </div>
  )
}

function CompareView({ selected }) {
  if (selected.length === 0) {
    return (
      <div className="empty-state">
        select two products from the products tab to compare
      </div>
    )
  }
  return (
    <div className="compare-panel">
      {selected.map((product) => {
        const previewPath = product.output_paths?.find((p) => p.endsWith('.png') || p.endsWith('.tif'))
        return (
          <div key={product.product_id} className="compare-slot">
            <img src={productImageUrl(previewPath)} alt={product.task} />
            <div className="slot-label">
              {product.task} · {product.completed_at?.slice(0, 10)}
            </div>
          </div>
        )
      })}
    </div>
  )
}

function TemporalView({ sceneId, products }) {
  const [index, setIndex] = useState(0)
  const sorted = [...products].sort((a, b) =>
    (a.completed_at || '').localeCompare(b.completed_at || '')
  )

  if (sorted.length === 0) {
    return <div className="empty-state">no products available for temporal view</div>
  }

  const current = sorted[index]
  const previewPath = current?.output_paths?.find((p) => p.endsWith('.png') || p.endsWith('.tif'))
  
  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', gap: 8 }}>
      <div style={{ flex: 1, display: 'flex', gap: 8 }}>
        <img
          src={productImageUrl(previewPath)}
          alt={current?.task}
          style={{ flex: 1, objectFit: 'contain', background: '#111' }}
        />
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '0 4px' }}>
        <button
          onClick={() => setIndex((i) => Math.max(0, i - 1))}
          disabled={index === 0}
          style={{ padding: '2px 8px', cursor: 'pointer' }}
        >
          ‹
        </button>
        <input
          type="range"
          min={0}
          max={sorted.length - 1}
          value={index}
          onChange={(e) => setIndex(Number(e.target.value))}
          style={{ flex: 1 }}
        />
        <button
          onClick={() => setIndex((i) => Math.min(sorted.length - 1, i + 1))}
          disabled={index === sorted.length - 1}
          style={{ padding: '2px 8px', cursor: 'pointer' }}
        >
          ›
        </button>
        <span style={{ fontSize: 11, color: '#aaa', minWidth: 80, textAlign: 'right' }}>
          {current?.task} {current?.completed_at?.slice(0, 10)}
        </span>
      </div>
    </div>
  )
}