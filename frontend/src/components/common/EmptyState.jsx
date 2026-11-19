import { Inbox } from 'lucide-react';

export function EmptyState({
  icon: Icon = Inbox,
  title = 'Nothing here yet',
  description,
  action,
}) {
  return (
    <div className="empty-state" role="status">
      <div className="empty-state-icon" aria-hidden="true">
        <Icon size={22} />
      </div>
      <h3 className="empty-state-title">{title}</h3>
      {description ? <p className="empty-state-desc">{description}</p> : null}
      {action ? <div>{action}</div> : null}
    </div>
  );
}

export default EmptyState;
