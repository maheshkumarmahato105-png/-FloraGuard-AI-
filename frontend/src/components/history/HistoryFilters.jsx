import { RotateCcw, Search } from 'lucide-react';

export function HistoryFilters({
  search,
  status,
  sortOrder,
  onSearchChange,
  onStatusChange,
  onSortChange,
  onReset,
}) {
  const hasFilters = Boolean(search || status !== 'all' || sortOrder !== 'newest');

  return (
    <section className="card" style={{ marginBottom: 20 }} aria-label="Filter prediction history">
      <div className="filters-bar">
        <div>
          <label
            htmlFor="history-search-input"
            style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: 6 }}
          >
            Search by disease or class
          </label>
          <div style={{ position: 'relative' }}>
            <Search
              size={15}
              aria-hidden="true"
              style={{
                position: 'absolute',
                left: 11,
                top: '50%',
                transform: 'translateY(-50%)',
                color: 'var(--text-muted)',
                pointerEvents: 'none',
              }}
            />
            <input
              id="history-search-input"
              type="search"
              className="input"
              style={{ paddingLeft: 34 }}
              placeholder="e.g. tomato, early_blight, healthy…"
              value={search}
              onChange={(e) => onSearchChange?.(e.target.value)}
            />
          </div>
        </div>

        <div>
          <label
            htmlFor="history-status-select"
            style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: 6 }}
          >
            Prediction status
          </label>
          <select
            id="history-status-select"
            className="select"
            value={status}
            onChange={(e) => onStatusChange?.(e.target.value)}
          >
            <option value="all">All statuses</option>
            <option value="success">Accepted (Success)</option>
            <option value="uncertain">Uncertain (Rejected)</option>
          </select>
        </div>

        <div>
          <label
            htmlFor="history-sort-select"
            style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: 6 }}
          >
            Sort order
          </label>
          <select
            id="history-sort-select"
            className="select"
            value={sortOrder}
            onChange={(e) => onSortChange?.(e.target.value)}
          >
            <option value="newest">Newest first</option>
            <option value="oldest">Oldest first</option>
          </select>
        </div>

        <div>
          <button
            type="button"
            className="btn btn-secondary"
            style={{ width: '100%' }}
            disabled={!hasFilters}
            onClick={onReset}
          >
            <RotateCcw size={14} aria-hidden="true" />
            Reset
          </button>
        </div>
      </div>
    </section>
  );
}

export default HistoryFilters;
