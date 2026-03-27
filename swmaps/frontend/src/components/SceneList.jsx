function ProductBadges({ products, tasks }) {
  const byTask = {}
  for (const p of products || []) {
    byTask[p.task] = p.status
  }
  if (!tasks?.length) return null 
  return (
    <div className="product-badges">
      {tasks.map((task) => {
        const status = byTask[task]
        if (!status) return (
          <span key={task} className="badge missing">{task}</span>
        )
        return (
          <span key={task} className={`badge ${status}`}>
            {task}
          </span>
        )
      })}
    </div>
  )
}

export default function SceneList({
  scenes,
  products,
  selectedSceneId,
  onSelectScene,
  compareMode,
  onToggleCompare,
  tasks,
}) {
  return (
    <div className="scene-pane">
      <div className="scene-list-header">
        <span>{scenes.length} scenes</span>
        <button
          className={compareMode ? 'active' : ''}
          onClick={onToggleCompare}
        >
          {compareMode ? 'exit compare' : 'compare'}
        </button>
      </div>
      <div className="scene-items">
        {scenes.length === 0 && (
          <div className="empty-state">
            Draw a bounding box or search to find scenes
          </div>
        )}
        {scenes.map((scene) => (
          <div
            key={scene.scene_id}
            className={`scene-item ${scene.scene_id === selectedSceneId ? 'selected' : ''}`}
            onClick={() => onSelectScene(scene.scene_id)}
          >
            <div className="scene-id">{scene.scene_id}</div>
            <div className="scene-meta">
              <span>{scene.sensor}</span>
              <span>{scene.acquisition_date}</span>
              <span>{scene.status}</span>
            </div>
            <ProductBadges products={products[scene.scene_id]} tasks={tasks} />
          </div>
        ))}
      </div>
    </div>
  )
}