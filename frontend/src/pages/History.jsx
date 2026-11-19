import { useCallback, useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { ChevronLeft, ChevronRight, History as HistoryIcon, RefreshCw, ScanSearch } from 'lucide-react';
import PageContainer from '../components/layout/PageContainer';
import HistoryFilters from '../components/history/HistoryFilters';
import HistoryTable from '../components/history/HistoryTable';
import ConfirmModal from '../components/common/ConfirmModal';
import EmptyState from '../components/common/EmptyState';
import ErrorMessage from '../components/common/ErrorMessage';
import LoadingSpinner from '../components/common/LoadingSpinner';
import { deleteHistoryRecord, listHistory } from '../services/historyService';
import { useToast } from '../context/ToastContext';
import { formatClassLabel } from '../utils/formatters';

const PAGE_SIZE = 10;

export function History() {
  const { addToast } = useToast();

  const [search, setSearch] = useState('');
  const [debouncedSearch, setDebouncedSearch] = useState('');
  const [status, setStatus] = useState('all');
  const [sortOrder, setSortOrder] = useState('newest');
  const [page, setPage] = useState(0);

  const [items, setItems] = useState([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const [pendingDelete, setPendingDelete] = useState(null);
  const [deleting, setDeleting] = useState(false);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      setDebouncedSearch(search);
      setPage(0);
    }, 220);
    return () => window.clearTimeout(timer);
  }, [search]);

  const loadHistory = useCallback(
    async (signal) => {
      setLoading(true);
      setError(null);
      try {
        const res = await listHistory({
          limit: PAGE_SIZE,
          offset: page * PAGE_SIZE,
          status,
          predictedClass: debouncedSearch,
          signal,
        });
        setItems(res?.items ?? []);
        setTotal(res?.total ?? 0);
      } catch (err) {
        if (err?.errorCode === 'aborted') return;
        setItems([]);
        setTotal(0);
        setError(err?.message || 'Failed to load prediction history.');
      } finally {
        setLoading(false);
      }
    },
    [debouncedSearch, page, status]
  );

  useEffect(() => {
    const controller = new AbortController();
    loadHistory(controller.signal);
    return () => controller.abort();
  }, [loadHistory]);

  const sortedRecords = useMemo(() => {
    const copy = [...items];
    copy.sort((a, b) => {
      const ta = new Date(a.created_at || 0).getTime();
      const tb = new Date(b.created_at || 0).getTime();
      return sortOrder === 'oldest' ? ta - tb : tb - ta;
    });
    return copy;
  }, [items, sortOrder]);

  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const hasFilters = Boolean(debouncedSearch || status !== 'all');

  const handleResetFilters = () => {
    setSearch('');
    setDebouncedSearch('');
    setStatus('all');
    setSortOrder('newest');
    setPage(0);
  };

  const handleConfirmDelete = async () => {
    if (!pendingDelete || deleting) return;
    setDeleting(true);
    try {
      await deleteHistoryRecord(pendingDelete.request_id);
      addToast({
        type: 'success',
        title: 'History record deleted',
        message: 'The record and any associated explanation artifacts were removed.',
      });
      setPendingDelete(null);

      // Step back a page if we just removed the last remaining item on a later page
      if (items.length === 1 && page > 0) {
        setPage((prev) => prev - 1);
      } else {
        loadHistory();
      }
    } catch (err) {
      addToast({
        type: 'error',
        title: 'Deletion failed',
        message: err?.message || 'Unable to delete the history record.',
      });
    } finally {
      setDeleting(false);
    }
  };

  return (
    <PageContainer
      title="Prediction History"
      subtitle="Inspect, filter, and manage real plant disease predictions stored in the backend SQLite database."
      actions={
        <>
          <button type="button" className="btn btn-secondary btn-sm" onClick={() => loadHistory()}>
            <RefreshCw size={14} aria-hidden="true" />
            Refresh
          </button>
          <Link to="/detection" className="btn btn-primary btn-sm">
            <ScanSearch size={14} aria-hidden="true" />
            New Scan
          </Link>
        </>
      }
    >
      <HistoryFilters
        search={search}
        status={status}
        sortOrder={sortOrder}
        onSearchChange={setSearch}
        onStatusChange={(val) => {
          setStatus(val);
          setPage(0);
        }}
        onSortChange={setSortOrder}
        onReset={handleResetFilters}
      />

      <section className="card" aria-label="Prediction history records">
        {loading ? (
          <LoadingSpinner label="Loading prediction history…" />
        ) : error ? (
          <ErrorMessage
            title="Unable to load history"
            message={error}
            recoveryHint="Confirm the Phase 7 FastAPI backend is running and click Try again."
            onRetry={() => loadHistory()}
          />
        ) : sortedRecords.length === 0 ? (
          <EmptyState
            icon={HistoryIcon}
            title={hasFilters ? 'No matching predictions found' : 'No prediction history yet'}
            description={
              hasFilters
                ? 'Try clearing your disease search term or changing the status filter.'
                : 'Once plant leaf images are analyzed by the backend, real prediction records will appear here.'
            }
            action={
              hasFilters ? (
                <button type="button" className="btn btn-secondary btn-sm" onClick={handleResetFilters}>
                  Clear Filters
                </button>
              ) : (
                <Link to="/detection" className="btn btn-primary btn-sm">
                  Start First Scan
                </Link>
              )
            }
          />
        ) : (
          <>
            <HistoryTable
              records={sortedRecords}
              onRequestDelete={(rec) => setPendingDelete(rec)}
            />

            <div className="pagination-bar">
              <span className="tabular">
                Showing {page * PAGE_SIZE + 1}–{Math.min((page + 1) * PAGE_SIZE, total)} of {total}{' '}
                {total === 1 ? 'record' : 'records'}
              </span>

              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <button
                  type="button"
                  className="btn btn-secondary btn-sm"
                  disabled={page === 0}
                  onClick={() => setPage((p) => Math.max(0, p - 1))}
                >
                  <ChevronLeft size={14} aria-hidden="true" />
                  Previous
                </button>
                <span className="tabular" style={{ padding: '0 6px' }}>
                  Page {page + 1} of {totalPages}
                </span>
                <button
                  type="button"
                  className="btn btn-secondary btn-sm"
                  disabled={page + 1 >= totalPages}
                  onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
                >
                  Next
                  <ChevronRight size={14} aria-hidden="true" />
                </button>
              </div>
            </div>
          </>
        )}
      </section>

      <ConfirmModal
        open={Boolean(pendingDelete)}
        title="Delete prediction record?"
        description={
          pendingDelete
            ? `Permanently delete the record for "${
                pendingDelete.status === 'uncertain'
                  ? 'Uncertain Sample'
                  : formatClassLabel(pendingDelete.predicted_class)
              }" (${pendingDelete.request_id}) and any linked Grad-CAM files? This action cannot be undone.`
            : ''
        }
        confirmLabel="Delete Record"
        confirmVariant="danger"
        busy={deleting}
        onConfirm={handleConfirmDelete}
        onCancel={() => setPendingDelete(null)}
      />
    </PageContainer>
  );
}

export default History;
